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
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)

        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current

        # Move all modules to the same device
        self.modules.to(self.device)
        # Move normalize statistics to the same device
        if hasattr(self.modules.normalize, "glob_mean"):
            self.modules.normalize.glob_mean = self.modules.normalize.glob_mean.to(self.device)
        if hasattr(self.modules.normalize, "glob_std"):
            self.modules.normalize.glob_std = self.modules.normalize.glob_std.to(self.device)

        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                feats = self.hparams.augmentation(feats)

        # forward modules
        src = self.modules.CNN(feats)

        enc_out, pred = self.modules.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index,
        )

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # Compute outputs
        hyps = None
        if stage == sb.Stage.TRAIN :
            hyps = None
        elif stage == sb.Stage.VALID:
            hyps = None
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % self.hparams.valid_search_interval == 0:
                # for the sake of efficiency, we only perform beamsearch with limited capacity
                # and no LM to give user some idea of how the AM is doing
                hyps, _ = self.hparams.valid_search(enc_out.detach(), wav_lens)
        elif stage == sb.Stage.TEST:
            hyps, _ = self.hparams.test_search(enc_out.detach(), wav_lens)
        return p_ctc, p_seq, wav_lens, hyps

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
    train_loader_kwargs={}):
    """
    Function that calculates the characteristics with improved error handling
    """
    measurements = {
        'Entropy mean': [], 'Entropy max': [], 'Entropy min': [], 'Entropy median': [],
        'Max mean': [], 'Max max': [], 'Max min': [], 'Max median': [],
        'Min mean': [], 'Min max': [], 'Min min': [], 'Min median': [],
        'Median mean': [], 'Median max': [], 'Median min': [], 'Median median': [],
        'JSD mean': [], 'JSD max': [], 'JSD min': [], 'JSD median': [],
        'KLD mean': [], 'KLD max': [], 'KLD min': [], 'KLD median': []
    }
    
    if progressbar is None:
        progressbar = not self.noprogressbar

    if not isinstance(train_set, (DataLoader, LoopedLoader)):
        train_loader_kwargs["ckpt_prefix"] = None
        train_set = self.make_dataloader(
            train_set, stage=sb.Stage.TEST, **train_loader_kwargs
        )
    
    self.on_evaluate_start(max_key=max_key, min_key=min_key)
    self.on_stage_start(sb.Stage.TEST, epoch=None)
    self.modules.eval()
    
    with torch.no_grad():
        for batch in tqdm(train_set, dynamic_ncols=True, disable=not progressbar):
            predictions = self.compute_forward(batch, stage=sb.Stage.TEST)
            p_ctc = torch.squeeze(predictions[0], dim=0)
            p_ctc_prob = torch.exp(p_ctc).detach().cpu().numpy()
            
            # Add small epsilon to avoid numerical issues
            epsilon = 1e-10
            p_ctc_prob = np.clip(p_ctc_prob, epsilon, 1.0 - epsilon)
            
            # Normalize probabilities
            p_ctc_prob = p_ctc_prob / p_ctc_prob.sum(axis=1, keepdims=True)
            
            try:
                # Entropy
                entropy_vals = -np.sum(p_ctc_prob * np.log(p_ctc_prob), axis=1)
                measurements['Entropy mean'].append(np.mean(entropy_vals))
                measurements['Entropy max'].append(np.max(entropy_vals))
                measurements['Entropy min'].append(np.min(entropy_vals))
                measurements['Entropy median'].append(np.median(entropy_vals))
                
                # Max probability
                max_prob = np.max(p_ctc_prob, axis=1)
                measurements['Max mean'].append(np.mean(max_prob))
                measurements['Max max'].append(np.max(max_prob))
                measurements['Max min'].append(np.min(max_prob))
                measurements['Max median'].append(np.median(max_prob))
                
                # Min probability
                min_prob = np.min(p_ctc_prob, axis=1)
                measurements['Min mean'].append(np.mean(min_prob))
                measurements['Min max'].append(np.max(min_prob))
                measurements['Min min'].append(np.min(min_prob))
                measurements['Min median'].append(np.median(min_prob))
                
                # Median probability
                median_prob = np.median(p_ctc_prob, axis=1)
                measurements['Median mean'].append(np.mean(median_prob))
                measurements['Median max'].append(np.max(median_prob))
                measurements['Median min'].append(np.min(median_prob))
                measurements['Median median'].append(np.median(median_prob))
                
                # JSD
                jsds = []
                for i in range(p_ctc_prob.shape[0] - 1):
                    p = p_ctc_prob[i]
                    q = p_ctc_prob[i + 1]
                    m = 0.5 * (p + q)
                    jsd = 0.5 * (np.sum(p * np.log(p/m)) + np.sum(q * np.log(q/m)))
                    jsds.append(jsd)
                
                if jsds:
                    measurements['JSD mean'].append(np.mean(jsds))
                    measurements['JSD max'].append(np.max(jsds))
                    measurements['JSD min'].append(np.min(jsds))
                    measurements['JSD median'].append(np.median(jsds))
                
                # KLD
                klds = []
                for i in range(p_ctc_prob.shape[0] - 1):
                    p = p_ctc_prob[i]
                    q = p_ctc_prob[i + 1]
                    kld = np.sum(p * np.log(p/q))
                    klds.append(kld)
                
                if klds:
                    measurements['KLD mean'].append(np.mean(klds))
                    measurements['KLD max'].append(np.max(klds))
                    measurements['KLD min'].append(np.min(klds))
                    measurements['KLD median'].append(np.median(klds))
                
            except Exception as e:
                print(f"Error in batch: {str(e)}")
                continue
    
    # Save measurements
    with open(path_file, 'wb') as file:
        pickle.dump(measurements, file, protocol=pickle.HIGHEST_PROTOCOL)


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
    
