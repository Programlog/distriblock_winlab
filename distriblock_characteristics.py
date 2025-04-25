"""
Script for computing Characteristics of the distribution over the output tokens using a Transformer-based ASR system with librispeech dataset.
The ASR system employs an encoder, a decoder, and an attention mechanism between them. 
Decoding is performed with (CTC/Att joint) beamsearch coupled with a neural language model.

Distriblock paper:
(https://arxiv.org/abs/2305.17000)

ASR model based on the SpeechBrain Transformer:
(https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech)
"""
import os
import sys
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader
from tqdm.contrib import tqdm
import numpy as np
from scipy.special import rel_entr
from scipy.spatial import distance
from scipy.stats import entropy
import pickle


logger = logging.getLogger(__name__)

# Define training procedure
class ASR(sb.core.Brain):
def compute_characteristics(self, wavs, wav_lens, tokens_bos, measurements):
    """
    Computes characteristics metrics for the given input data.

    Args:
        wavs (torch.Tensor): Input waveforms.
        wav_lens (torch.Tensor): Lengths of waveforms as a proportion of the max length.
        tokens_bos (torch.Tensor): Tokens with beginning-of-sequence marker.
        measurements (dict): Dictionary to store computed metrics.

    Returns:
        None
    """
    try:
        # Compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current

        # Move modules’ normalize statistics to the correct device if needed
        self.modules.to(self.device)
        if hasattr(self.modules.normalize, "glob_mean"):
            self.modules.normalize.glob_mean = self.modules.normalize.glob_mean.to(self.device)
        if hasattr(self.modules.normalize, "glob_std"):
            self.modules.normalize.glob_std = self.modules.normalize.glob_std.to(self.device)

        # Normalize features
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        # Forward pass through modules
        src = self.modules.CNN(feats)
        enc_out, pred = self.modules.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index,
        )
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        # Convert log probabilities to probabilities safely
        epsilon = 1e-10  # Small constant to avoid numerical issues
        p_ctc_np = torch.exp(p_ctc).detach().cpu().numpy()
        p_ctc_np = np.clip(p_ctc_np, epsilon, None)  # Ensure no zeros

        # Normalize probabilities row-wise
        row_sums = np.sum(p_ctc_np, axis=1, keepdims=True)
        mask = row_sums < epsilon
        if np.any(mask):
            # Replace invalid rows with uniform distribution
            uniform_prob = 1.0 / p_ctc_np.shape[1]
            p_ctc_np[mask.flatten()] = uniform_prob
            row_sums = np.sum(p_ctc_np, axis=1, keepdims=True)

        p_ctc_np = p_ctc_np / row_sums  # Normalize rows
        p_ctc_np = np.clip(p_ctc_np, epsilon, 1.0 - epsilon)  # Avoid log(0)

        # Compute characteristics
        entropy_vals = -np.sum(p_ctc_np * np.log(p_ctc_np), axis=1)
        measurements['Entropy mean'].append(np.mean(entropy_vals))
        measurements['Entropy max'].append(np.max(entropy_vals))
        measurements['Entropy min'].append(np.min(entropy_vals))
        measurements['Entropy median'].append(np.median(entropy_vals))

        max_prob = np.max(p_ctc_np, axis=1)
        measurements['Max mean'].append(np.mean(max_prob))
        measurements['Max max'].append(np.max(max_prob))
        measurements['Max min'].append(np.min(max_prob))
        measurements['Max median'].append(np.median(max_prob))

        min_prob = np.min(p_ctc_np, axis=1)
        measurements['Min mean'].append(np.mean(min_prob))
        measurements['Min max'].append(np.max(min_prob))
        measurements['Min min'].append(np.min(min_prob))
        measurements['Min median'].append(np.median(min_prob))

        median_prob = np.median(p_ctc_np, axis=1)
        measurements['Median mean'].append(np.mean(median_prob))
        measurements['Median max'].append(np.max(median_prob))
        measurements['Median min'].append(np.min(median_prob))
        measurements['Median median'].append(np.median(median_prob))

        # JSD computation
        jsds = []
        for i in range(p_ctc_np.shape[0] - 1):
            p = p_ctc_np[i]
            q = p_ctc_np[i + 1]
            m = 0.5 * (p + q)
            jsd = 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
            jsds.append(jsd)
        if jsds:
            measurements['JSD mean'].append(np.mean(jsds))
            measurements['JSD max'].append(np.max(jsds))
            measurements['JSD min'].append(np.min(jsds))
            measurements['JSD median'].append(np.median(jsds))

        # KLD computation
        klds = []
        for i in range(p_ctc_np.shape[0] - 1):
            p = p_ctc_np[i]
            q = p_ctc_np[i + 1]
            kld = np.sum(p * np.log(p / q))
            klds.append(kld)
        if klds:
            measurements['KLD mean'].append(np.mean(klds))
            measurements['KLD max'].append(np.max(klds))
            measurements['KLD min'].append(np.min(klds))
            measurements['KLD median'].append(np.median(klds))

    except Exception as e:
        print(f"Error during computation: {str(e)}")

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        """ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)"""
        self.hparams.model.eval()
        print("Loaded the average")

    def no_checkpoint_average(self, max_key=None, min_key=None):
         """Skip checkpoint averaging and use the pre-trained model directly"""
         super().on_evaluate_start()
         print("Skipping checkpoint averaging, using pre-trained model directly")

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def characteristics(
        self,
        train_set,
        path_file, 
        max_key=None,
        min_key=None,            
        hparams=None,
        progressbar=None,
        train_loader_kwargs={},):
        """
        Function that calculates the 24 scores 
        (resulting from combing each of the 4 aggregation methods with the 6 characteristics).
        """
        
        # Characteristics
        measurements = {
            'Entropy mean': 0, 'Entropy max': 0, 'Entropy min': 0, 'Entropy median': 0,
            'Max mean': 0, 'Max max': 0, 'Max min': 0, 'Max median': 0,
            'Min mean': 0, 'Min max': 0, 'Min min': 0, 'Min median': 0,
            'Median mean': 0, 'Median max': 0, 'Median min': 0, 'Median median': 0,
            'JSD mean': 0, 'JSD max': 0, 'JSD min': 0, 'JSD median': 0, 
            'KLD mean': 0, 'KLD max': 0, 'KLD min': 0, 'KLD median': 0
            }
        
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (
                isinstance(train_set, DataLoader)
                or isinstance(train_set, LoopedLoader)
        ):
            train_loader_kwargs["ckpt_prefix"] = None
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TEST, **train_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()  

        entr_avg, entr_max, entr_min, entr_med = [], [], [], []
        max_avg, max_max, max_min, max_med = [], [], [], []
        min_avg, min_max, min_min, min_med = [], [], [], []
        med_avg, med_max, med_min, med_med = [], [], [], []
        jsd_avg, jsd_max, jsd_min, jsd_med = [], [], [], []  
        kld_avg, kld_max, kld_min, kld_med = [], [], [], []     

        with torch.no_grad():
            for batch in tqdm(train_set, dynamic_ncols=True, disable=not progressbar):
                predictions = self.compute_forward(batch, stage=sb.Stage.TEST)
                p_ctc = torch.squeeze(predictions[0], dim=0)
                p_ctc_prob = torch.exp(p_ctc).detach().cpu()
                
                p_ctc_prob = np.array(p_ctc_prob)
                # Remove extreme cases which lead to undefined characteristic values
                p_ctc_prob = np.delete(p_ctc_prob, np.where((p_ctc_prob == 0))[0], axis=0)    
                p_ctc_prob = np.delete(p_ctc_prob, np.where((p_ctc_prob == 1))[0], axis=0)   
                # Entropy
                entropy_1 = entropy(p_ctc_prob, axis=1)
                entr_avg.append(np.mean(entropy_1))
                entr_max.append(np.max(entropy_1))
                entr_min.append(np.min(entropy_1))
                entr_med.append(np.median(entropy_1))
                # Max
                max_prob = np.max(p_ctc_prob, axis=1)
                max_avg.append(np.mean(max_prob))
                max_max.append(np.max(max_prob))
                max_min.append(np.min(max_prob))
                max_med.append(np.median(max_prob))
                # Min
                min_prob = np.log(np.min(p_ctc_prob, axis=1))
                min_avg.append(np.mean(min_prob))
                min_max.append(np.max(min_prob))
                min_min.append(np.min(min_prob))
                min_med.append(np.median(min_prob))
                # Median
                median_prob = np.log(np.median(p_ctc_prob, axis=1))
                med_avg.append(np.mean(median_prob))
                med_max.append(np.max(median_prob))
                med_min.append(np.min(median_prob))
                med_med.append(np.median(median_prob))
                # JSD Divergense distance
                d_sym, kl_sym = [], []
                for i in range(p_ctc_prob.shape[0] - 1):
                    m_2 = 0.5 * (p_ctc_prob[i, :] + p_ctc_prob[i + 1, :])
                    left = 0.5 * np.sum(rel_entr(p_ctc_prob[i, :], m_2))
                    right = 0.5 * np.sum(rel_entr(p_ctc_prob[i + 1, :], m_2))    
                    jsd = left + right 
                    d_sym.append(jsd)
                    kl_sym.append(np.sum(rel_entr(p_ctc_prob[i, :], p_ctc_prob[i + 1, :])))           
                d_sym = np.array(d_sym)
                jsd_avg.append(np.mean(d_sym))
                jsd_max.append(np.max(d_sym))
                jsd_min.append(np.min(d_sym))
                jsd_med.append(np.median(d_sym))

                kl_sym = np.array(kl_sym)
                kld_avg.append(np.mean(kl_sym))
                kld_max.append(np.max(kl_sym))
                kld_min.append(np.min(kl_sym))
                kld_med.append(np.median(kl_sym))
        
        # Saving the Characteristics
        measurements['Entropy mean'], measurements['Entropy max'], measurements['Entropy min'], \
        measurements['Entropy median'] = entr_avg, entr_max, entr_min, entr_med
        measurements['Max mean'], measurements['Max max'], measurements['Max min'], measurements['Max median'] = \
            max_avg, max_max, max_min, max_med
        measurements['Min mean'], measurements['Min max'], measurements['Min min'], measurements['Min median'] = \
            min_avg, min_max, min_min, min_med
        measurements['Median mean'], measurements['Median max'], measurements['Median min'], measurements['Median median'] \
            = med_avg, med_max, med_min, med_med
        measurements['JSD mean'], measurements['JSD max'], measurements['JSD min'], \
        measurements['JSD median'] = jsd_avg, jsd_max, jsd_min, jsd_med
        measurements['KLD mean'], measurements['KLD max'], measurements['KLD min'], \
        measurements['KLD median'] = kld_avg, kld_max, kld_min, kld_med
        with open(path_file, 'wb') as file:
            pickle.dump(measurements, file, protocol=pickle.HIGHEST_PROTOCOL)
        pass

def dataio_prepare(hparams, file_path):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=file_path
    )
    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "random":
        pass
    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
    datasets = [train_data]
    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig", "path")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        yield sig
        yield wav

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens", "path"],
    )
    return (train_data)

if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If --distributed_launch then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    asr_brain.tokenizer = hparams["tokenizer"]

    train_set = dataio_prepare(hparams, hparams["train_transcriptions"])
    val_set = dataio_prepare(hparams, hparams["val_transcriptions"])
    test_set = dataio_prepare(hparams, hparams["test_transcriptions"])
    adv_val = dataio_prepare(hparams, hparams["adv_train_transcription"])
    adv_test = dataio_prepare(hparams, hparams["adv_test_transcriptions"])

    file_names = ["train.pickle", "val.pickle", "test.pickle", "adv_train.pickle", "adv_test.pickle", ]

    data_sets = [train_set, val_set, test_set, adv_val, adv_test]
    characteristic_folder = hparams["characteristic_folder"]
    attack_type = hparams["attack_type"]
    if not os.path.exists(characteristic_folder):
        os.makedirs(characteristic_folder)
    if not os.path.exists(f"{characteristic_folder}/{attack_type}"):
        os.makedirs(f"{characteristic_folder}/{attack_type}")

    for i, data in enumerate(data_sets):
        if i > 2:
            root_path = f"{characteristic_folder}/{attack_type}"
        else:
            root_path = hparams["characteristic_folder"]
        if(not os.path.exists(f"{root_path}/{file_names[i]}")):
            print(f"Saving characteristics in file: {file_names[i]}!")
            asr_brain.characteristics(
                data, 
                f"{root_path}/{file_names[i]}", 
                max_key="ACC",
                train_loader_kwargs=hparams["test_dataloader_opts"])
    
