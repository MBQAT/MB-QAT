import sys
import timeit
from pathlib import Path
import os

import h5py
import soundfile as sf
import torch
import torch.nn as nn
import yaml
import torch.quantization.quantize_fx as quantize_fx
import copy

from Checkpoint_MBQAT import CheckpointMB
from learn_scale_MBQAT import prepare
from net_100k_MBQAT import Net
from torch.autograd import Variable
from thop import profile, clever_format
from tqdm import tqdm
from pystoi import stoi
from pypesq import pesq
from CRN_original.dataloader import TrainDataset, TrainDataLoader

sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
from utils.stft import STFT
from utils.util import normalize_wav, gen_list, snr
from utils.ConfigArgs import ConfigArgs
from utils.progressbar import progressbar as pb


def printResult(dict, metric_type):
    noises = ['ADTbabble', 'ADTcafeteria']
    snrs = ['snr-5', 'snr-2', 'snr0', 'snr2', 'snr5']
    print(metric_type, end='\t')
    for n in noises:
        for r in snrs:
            domain = n + '_' + r
            print('(' + domain + ')', end='\t')
    print()
    print('MIX ', end='\t')
    for n in noises:
        for r in snrs:
            domain = n + '_' + r
            print(round(dict[domain + '_mix'], 4), end='\t')
    print()
    print('TIME', end='\t')
    for n in noises:
        for r in snrs:
            domain = n + '_' + r
            print(round(dict[domain + '_time'], 4), end='\t')
    print()


def quantize_model(model, mode, data_args=None):
    if mode == 'dynamic':
        model_int8 = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.LSTM},
            dtype=torch.qint8)
        return model_int8

    if mode == 'static':

        wav_data, sr = sf.read(data_args.test_wav)
        wav_data = torch.Tensor(wav_data).squeeze().unsqueeze(0)
        torch_stft = STFT(320, 160)
        mix_mag, mix_pha = torch_stft.stft(wav_data)
        example_inputs = (mix_mag)

        from torch.ao.quantization import get_default_qconfig_mapping
        qconfig_mapping = get_default_qconfig_mapping()

        model_prepared = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs)

        data_args.num_train = 3200 * 5
        data_train = TrainDataset(data_args)
        tr_batch_dataloader = TrainDataLoader(data_train, 32, is_shuffle=True, workers_num=8)

        for i_batch, batch_data in tqdm(enumerate(tr_batch_dataloader.get_dataloader())):
            src_mag, src_pha = torch_stft.stft(batch_data[0])
            model_prepared(src_mag)

        quan_checkpoint = CheckpointMB(checkpoint.start_epoch, checkpoint.train_loss, best_loss,
                                       model_prepared.state_dict(), checkpoint.optimizer)
        quan_checkpoint.save(False, model_path + save_path + '100k-e100-s.ckpt')

        model_after_quantized = quantize_fx.convert_fx(model_prepared)
        return model_after_quantized


if __name__ == '__main__':

    with open('config_100k_MBQAT.yaml', 'r') as f_yaml:
        config = yaml.load(f_yaml, Loader=yaml.FullLoader)
    _abspath = Path(os.path.abspath(__file__)).parent
    config['project'] = _abspath.parent.stem
    config['workspace'] = _abspath.stem
    args = ConfigArgs(config)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_ids

    model_path = args.output_dir
    load_path = args.PTQ_load_path
    save_path = args.PTQ_save_path

    net = Net()

    qat_config = {'a_bit': 8, 'w_bit': 4, "all_positive": False, "per_channel": False,
                  "batch_init": 0}
    prepare(net, inplace=True, a_bits=qat_config["a_bit"], w_bits=qat_config["w_bit"],
            all_positive=qat_config["all_positive"], per_channel=qat_config["per_channel"],
            batch_init=qat_config["batch_init"])

    checkpoint = CheckpointMB()
    checkpoint.load(model_path + load_path + 'best.ckpt')
    best_loss = checkpoint.best_loss
    net.load_state_dict(checkpoint.state_dict)

    model_to_quantize = copy.deepcopy(net)
    model_to_quantize.eval()

    rod_input = Variable(torch.FloatTensor(torch.rand(1, 99, 161)))
    macs, params = profile(net, inputs=(rod_input,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs)
    print(params)

    model_d = quantize_model(model_to_quantize, 'dynamic')

    samp_list = gen_list(args.test_path, '.samp')

    torch_stft = STFT(args.frame_size, args.frame_shift)

    score_stois = {}
    score_snrs = {}
    score_pesqs = {}

    for i in range(len(samp_list)):
        filename_input = samp_list[i]
        elements = filename_input.split('_')
        noise_type, snr_value = elements[1], elements[2]
        print('{}/{}, Started working on {}.'.format(i + 1, len(samp_list), samp_list[i]))
        f_mix = h5py.File(os.path.join(args.test_path, filename_input), 'r')
        ttime, mtime, cnt = 0., 0., 0.
        acc_stoi_mix, acc_snr_mix, acc_pesq_mix = 0., 0., 0.
        acc_stoi_time, acc_snr_time, acc_pesq_time = 0., 0., 0.
        num_clips = len(f_mix)
        ttbar = pb(0, num_clips, 20)
        ttbar.start()
        for k in range(num_clips):
            start = timeit.default_timer()
            reader_grp = f_mix[str(k)]
            mix = reader_grp['noisy_raw'][:]
            label = reader_grp['clean_raw'][:]
            mix_mag, mix_pha = torch_stft.stft(torch.from_numpy(mix).reshape(1, -1))

            est_mask = model_d(mix_mag)
            est = mix_mag * est_mask

            real = est * torch.cos(mix_pha)
            imag = est * torch.sin(mix_pha)
            est_time = torch_stft.istft(torch.stack([real, imag], 1))

            est_time = est_time.detach().numpy()[0]
            mix = mix[:est_time.size]
            label = label[:est_time.size]

            mix_stoi = stoi(label, mix, 16000)
            est_stoi_time = stoi(label, est_time, 16000)
            acc_stoi_mix += mix_stoi
            acc_stoi_time += est_stoi_time

            mix_snr = snr(label, mix)
            est_snr_time = snr(label, est_time)
            acc_snr_mix += mix_snr
            acc_snr_time += est_snr_time

            mix_pesq = pesq(label, mix, 16000)
            est_pesq_time = pesq(label, est_time, 16000)
            acc_pesq_mix += mix_pesq
            acc_pesq_time += est_pesq_time

            cnt += 1
            end = timeit.default_timer()
            curr_time = end - start
            ttime += curr_time
            mtime = ttime / cnt

            ttbar.update_progress(k, 'test', 'ctime/mtime={:.3f}/{:.3f}'.format(curr_time, mtime))

            label_norm, label_scale = normalize_wav(label)
            mix_norm, mix_scale = normalize_wav(mix)
            est_time_norm, time_scale = normalize_wav(est_time)

            sf.write('%sS%.3d_%s_%s_mix.wav' % (args.prediction_path, k, noise_type, snr_value), mix_norm,
                     16000)
            sf.write('%sS%.3d_%s_%s_tgt.wav' % (args.prediction_path, k, noise_type, snr_value), label_norm,
                     16000)
            sf.write('%sS%.3d_%s_%s_time.wav' % (args.prediction_path, k, noise_type, snr_value),
                     est_time_norm, 16000)

        score_stois[noise_type + '_' + snr_value + '_mix'] = acc_stoi_mix / num_clips
        score_stois[noise_type + '_' + snr_value + '_time'] = acc_stoi_time / num_clips
        score_snrs[noise_type + '_' + snr_value + '_mix'] = acc_snr_mix / num_clips
        score_snrs[noise_type + '_' + snr_value + '_time'] = acc_snr_time / num_clips
        score_pesqs[noise_type + '_' + snr_value + '_mix'] = acc_pesq_mix / num_clips
        score_pesqs[noise_type + '_' + snr_value + '_time'] = acc_pesq_time / num_clips
        ttbar.finish()
        f_mix.close()
    printResult(score_snrs, 'SNR ')
    printResult(score_stois, 'STOI')
    printResult(score_pesqs, 'PESQ')
