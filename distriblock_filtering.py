"""
Script for evaluating LPF+SG filtering to preserve system robustness.
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
from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader
from tqdm.contrib import tqdm
import numpy as np
import pickle
from scipy.signal import lfilter
import noisereduce as nr
from src.classifiers import gaussian_filtering
from src.tools import list_load_data, word_error_rate, character_error_rate
from speechbrain.utils.distributed import run_on_main


logger = logging.getLogger(__name__)

# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage, fil_type):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos

        if asr_brain.hparams.ratio == 0.5:
            if fil_type == "lpf":
                wavs = self.low_pass_filter(wavs).to(self.device)
            elif fil_type == "sg":
                wavs = torch.from_numpy(nr.reduce_noise(y=wavs.cpu(), sr=16000)).to(self.device)
            elif fil_type == "lpf-sg":
                wavs = self.low_pass_filter(wavs).to(self.device)
                wavs = torch.from_numpy(nr.reduce_noise(y=wavs.cpu(), sr=16000)).to(self.device)

        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        # forward modules
        src = self.modules.CNN(feats)

        enc_out, pred = self.modules.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index,
        )
        # Compute outputs
        hyps = None
        hyps, _ = self.hparams.test_search(enc_out.detach(), wav_lens)
        # Decode token terms to words
        predicted_words = [
            tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps
        ]
        return " ".join(predicted_words[0])

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()
        print("Loaded the average")

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def asr_output(
            self,
            train_set,
            max_key=None,
            min_key=None,            
            hparams=None,
            progressbar=None,
            train_loader_kwargs={},
            fil_type=None
    ):
       
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
        pred_transc = []
        with torch.no_grad():
            for batch in tqdm(train_set, dynamic_ncols=True,
                    disable=not progressbar,
                    colour=self.tqdm_barcolor["test"]):
                predictions = self.compute_forward(batch, sb.Stage.TEST, fil_type)
                pred_transc.append(predictions)

        return pred_transc

    def low_pass_filter(self, s1, fs=16000):    
        # Low pass filter coefficients, cut off frequency of 7000 Hz
        # Applying filter
        signal_lpf = torch.Tensor(lfilter(LPF_COEFFS, 1, s1.cpu()))
        return signal_lpf

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

def asr_performance(data_set, fil_type):
    asr_brain.hparams.ratio = 1
    stry  = asr_brain.asr_output(data_set, hparams=hparams, train_loader_kwargs=hparams["test_dataloader_opts"], fil_type=fil_type)

    asr_brain.hparams.ratio = 0.5
    halfy  = asr_brain.asr_output(data_set, hparams=hparams, train_loader_kwargs=hparams["test_dataloader_opts"], fil_type=fil_type)
   
    num_samples = len(stry)
    wer, cer = [], []
    for i in range(num_samples):
        if (stry[i]== "" or halfy[i]==""):
            print("Empty String index: ", i)
        else:
            wer.append(float(word_error_rate(stry[i], halfy[i])))
            cer.append(float(character_error_rate(stry[i], halfy[i])))
    return wer, cer

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
    tokenizer = hparams["tokenizer"]

    train_set = dataio_prepare(hparams, hparams["train_transcriptions"])
    val_set = dataio_prepare(hparams, hparams["val_transcriptions"])
    test_set = dataio_prepare(hparams, hparams["test_transcriptions"])
    adv_val = dataio_prepare(hparams, hparams["adv_train_transcription"])
    adv_test = dataio_prepare(hparams, hparams["adv_test_transcriptions"])

    data_sets = [train_set, val_set, test_set, adv_val, adv_test]
    wer_files = ["train_wer.pickle", "val_wer.pickle", "test_wer.pickle", "adv_val_wer.pickle", "adv_test_wer.pickle"]
    cer_files = ["train_cer.pickle", "val_cer.pickle", "test_cer.pickle", "adv_val_cer.pickle", "adv_test_cer.pickle"]

    characteristic_folder = hparams["characteristic_folder"]
    if not os.path.exists(f"{characteristic_folder}/filter"):
        os.makedirs(f"{characteristic_folder}/filter")

    fil_type = ["lpf", "sg", "lpf-sg"]
    LPF_COEFFS = np.load('lpf_coeff.npy')
    for i, f_nm in enumerate(fil_type):
        filter_folder = f"{characteristic_folder}/filter/{f_nm}"
        if not os.path.exists(filter_folder):
            os.makedirs(filter_folder)
        for k, data_ in enumerate(data_sets):
            if not os.path.exists(f"{filter_folder}/{wer_files[k]}") or True:
                print("Filter method: \"{}\". Calculating WER/CER ... ".format(f_nm))
                wer, cer = asr_performance(data_, f_nm)
                with open(f"{filter_folder}/{wer_files[k]}", 'wb') as file:
                    pickle.dump(wer, file, protocol=pickle.HIGHEST_PROTOCOL)
                with open(f"{filter_folder}/{cer_files[k]}", 'wb') as file:
                    pickle.dump(cer, file, protocol=pickle.HIGHEST_PROTOCOL)                
    
    bengign_flag = [True, False]
    for i, f_nm in enumerate(fil_type):
        filter_folder = f"{characteristic_folder}/filter/{f_nm}"
        # Evaluate gaussian classifier based on WER
        tra_wer = list_load_data(f"{filter_folder}/{wer_files[0]}", benign_flg=bengign_flag[0])
        val_wer = list_load_data(f"{filter_folder}/{wer_files[1]}", benign_flg=bengign_flag[0])
        test_wer = list_load_data(f"{filter_folder}/{wer_files[2]}", benign_flg=bengign_flag[0])
        adv_val_wer = list_load_data(f"{filter_folder}/{wer_files[3]}", benign_flg=bengign_flag[1])
        adv_test_wer = list_load_data(f"{filter_folder}/{wer_files[4]}", benign_flg=bengign_flag[1])
        metrics = gaussian_filtering(tra_wer, test_wer, val_wer, adv_val_wer, adv_test_wer)   
        print("Gaussian classifier based on WER metric using filtering method: \"{}\":".format(f_nm))
        print("Accuracy: {:.2%} TP: {} FP: {} TN: {} FN: {} FPR: {:.2f} TPR: {:.2f} precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5], metrics[6], metrics[7], metrics[8], metrics[9]))
        # Evaluate gaussian classifier based on CER
        tra_cer = list_load_data(f"{filter_folder}/{cer_files[0]}", benign_flg=bengign_flag[0])
        val_cer = list_load_data(f"{filter_folder}/{cer_files[1]}", benign_flg=bengign_flag[0])
        test_cer = list_load_data(f"{filter_folder}/{cer_files[2]}", benign_flg=bengign_flag[0])
        adv_val_cer = list_load_data(f"{filter_folder}/{cer_files[3]}", benign_flg=bengign_flag[1])
        adv_test_cer = list_load_data(f"{filter_folder}/{cer_files[4]}", benign_flg=bengign_flag[1])
        metrics = gaussian_filtering(tra_cer, test_cer, val_cer, adv_val_cer, adv_test_cer)   
        print("Gaussian classifier based on CER metric using filtering method: \"{}\":".format(f_nm))
        print("Accuracy: {:.2%} TP: {} FP: {} TN: {} FN: {} FPR: {:.2f} TPR: {:.2f} precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5], metrics[6], metrics[7], metrics[8], metrics[9]))
