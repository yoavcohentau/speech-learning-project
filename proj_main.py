import os

import numpy as np
import torch
from scipy.io import wavfile

from utils.speech_generator_utils import generate_room_impulse_responses, generate_microphone_signals,\
    generate_white_noise, mix_signals, plot_time_freq_analysis
from utils.mvdr_utils import apply_mvdr
from utils.metric_utils import AudioMetrics, parse_and_plot_results
from utils.librispeech_data_set_utils import load_librispeech_objects_from_yaml
from DeepTaylorBeamformer.nets.TaylorBeamformer import TaylorBeamformer
from DeepTaylorBeamformer.taylor_inference_func import apply_taylor_net, align_signal

PLOT_AND_SAVE_FLAG = False
EXAMPLE_IDX_TO_SAVE = 0
SNR_TO_SAVE = 0
T60_TO_SAVE = 0.3
ORIGINAL_SIGNAL_FACTOR = 0.7

# TODO: user - please change the following two lines to your LibriSpeech data path
DATA_SET_NAME = "dev-clean"  # "dev-clean" or "test-clean"
DATA_SET_PATH = fr"J:\My Drive\Courses\2026A\Signal Processing and Machine Learning for Speech\HW\HW1\SpeechLearningCourseEx1\data\{DATA_SET_NAME}\LibriSpeech"


def taylor_main():
    # Setup Parameters
    fs = 16000
    room_dim = [4, 5, 3]
    T60_vec = [0.15, 0.3]
    snr_vec = [0, 10]

    mic_center = np.array([2, 1, 1.7])
    num_mics = 5
    mic_spacing = 0.05

    # Mics Positions
    mic_positions = np.array([
        mic_center + np.array([(i - (num_mics - 1) / 2) * mic_spacing, 0, 0])
        for i in range(num_mics)
    ])

    # Source Parameters
    # Target
    src_theta = np.deg2rad(30)
    source_pos = mic_center + 1.5 * np.array([np.cos(src_theta), np.sin(src_theta), 0])

    # Interferer (for noise type 2)
    int_theta = np.deg2rad(150)
    interferer_pos = mic_center + 2.0 * np.array([np.cos(int_theta), np.sin(int_theta), 0])

    # Initialize Metrics Calculator
    metrics_tool = AudioMetrics(fs)
    all_metrics = []  # Store results for aggregation

    # load samples file names
    yaml_path = 'file_name_list.yaml'
    signal_objects, interferer_objects = load_librispeech_objects_from_yaml(
        yaml_path,
        DATA_SET_PATH,
        DATA_SET_NAME
    )

    # load checkpoints
    checkpoint_load_path = r".\DeepTaylorBeamformer\trained_model_weights"
    checkpoint_load_filename = r"best_e52_01_3_26.pth"
    checkpoint = torch.load(os.path.join(checkpoint_load_path, checkpoint_load_filename),
                            map_location=torch.device('cpu'))

    taylor_net = TaylorBeamformer(
        k1=[1, 3],
        k2=[2, 3],
        ref_mic=0,
        c=64,
        embed_dim=64,
        fft_num=320,
        order_num=3,
        kd1=5,
        cd1=64,
        d_feat=256,
        dilations=[1, 2, 5, 9],
        group_num=2,
        hid_node=64,
        M=5,
        rnn_type="LSTM",
        intra_connect="cat",
        inter_connect="cat",
        out_type="mapping",
        bf_type="embedding",
        norm2d_type="BN",
        norm1d_type="BN",
        is_compress=False,
        is_total_separate=False,
        is_u2=True,
        is_1dgate=True,
        is_squeezed=False,
        is_causal=True,
        is_param_share=False
    )
    taylor_net.load_state_dict(checkpoint)

    for T60 in T60_vec:
        for snr in snr_vec:
            for example_idx, (signal_object, interferer_object) in enumerate(zip(signal_objects, interferer_objects)):
                print(f'---------- example #{example_idx} ----------')

                metrics = {}

                # Target RIR
                target_rirs = generate_room_impulse_responses(fs, room_dim, mic_center, num_mics, mic_spacing, 30, 1.5, [T60])
                target_path = signal_object.params2path()
                target_sigs = generate_microphone_signals(target_path, fs, target_rirs)[T60]

                # Interferer RIR
                inter_rirs = generate_room_impulse_responses(fs, room_dim, mic_center, num_mics, mic_spacing, 150, 2.0, [T60])
                inter_path = interferer_object.params2path()
                inter_sigs = generate_microphone_signals(inter_path, fs, inter_rirs)[T60]

                # Cut to same length
                min_len = min(target_sigs.shape[1], inter_sigs.shape[1])
                target_sigs = target_sigs[:, :min_len]
                inter_sigs = inter_sigs[:, :min_len]

                # add Noise
                # Case A: White Noise
                white_noise = generate_white_noise(target_sigs.shape)
                noisy_white, white_noise_scaled = mix_signals(target_sigs, white_noise, snr)

                # Case B: Interferer
                noisy_interferer, inter_noise_scaled = mix_signals(target_sigs, inter_sigs, snr)

                ref_mic_index = 2  # Center mic
                ref_noisy_white = noisy_white[ref_mic_index]
                ref_noisy_inter = noisy_interferer[ref_mic_index]
                target_clean_ref = target_sigs[ref_mic_index]

                if PLOT_AND_SAVE_FLAG and example_idx == EXAMPLE_IDX_TO_SAVE and snr == SNR_TO_SAVE and T60 == T60_TO_SAVE:
                    # Save noisy signals
                    wavfile.write("output_folder_proj/white_in.wav", fs, ref_noisy_white.astype(np.float32))
                    wavfile.write("output_folder_proj/interferer_in.wav", fs, ref_noisy_inter.astype(np.float32))

                # --- (1) MVDR ---
                # Case 1: White Noise
                mvdr_white_out = apply_mvdr(noisy_white, white_noise_scaled)
                mvdr_white_out = mvdr_white_out[:min_len]

                # Case 2: Interferer
                mvdr_inter_out = apply_mvdr(noisy_interferer, inter_noise_scaled)
                mvdr_inter_out = mvdr_inter_out[:min_len]

                # --- Plot & Save ---
                # Save metrics
                metrics[f'MVDR-white-{snr}-{T60}-{example_idx}'] = metrics_tool.compute_all(target_clean_ref, mvdr_white_out)
                metrics[f'MVDR-inter-{snr}-{T60}-{example_idx}'] = metrics_tool.compute_all(target_clean_ref, mvdr_inter_out)

                if PLOT_AND_SAVE_FLAG and example_idx == EXAMPLE_IDX_TO_SAVE and snr == SNR_TO_SAVE and T60 == T60_TO_SAVE:
                    # White Noise
                    wavfile.write("output_folder_proj/mvdr_white_out.wav", fs, mvdr_white_out.astype(np.float32))
                    plot_time_freq_analysis(target_clean_ref/ORIGINAL_SIGNAL_FACTOR, ref_noisy_white, mvdr_white_out, fs,
                                            f"(MVDR Output - White Noise - T60={T60}s - snr={snr}dB)",
                                            "Original", "Noisy", "Beamformer Out")

                    # Interferer
                    wavfile.write("output_folder_proj/mvdr_interferer_out.wav", fs, mvdr_inter_out.astype(np.float32))
                    plot_time_freq_analysis(target_clean_ref/ORIGINAL_SIGNAL_FACTOR, ref_noisy_inter, mvdr_inter_out, fs,
                                            f"(MVDR Output - Interferer - T60={T60}s - snr={snr}dB)",
                                            "Original", "Noisy", "Beamformer Out")

                print("MVDR Done.")

                # --- (2) Deep Taylor ---
                # Case 1: White Noise
                taylor_white_out = apply_taylor_net(taylor_net, noisy_white, fs)
                taylor_white_out = taylor_white_out[:min_len]

                # Case 2: Interferer
                taylor_inter_out = apply_taylor_net(taylor_net, noisy_interferer, fs)
                taylor_inter_out = taylor_inter_out[:min_len]

                # --- Plot & Save ---
                # Save metrics
                target_clean_ref_1, taylor_white_out_1 = align_signal(target_clean_ref, taylor_white_out)
                metrics[f'Taylor-white-{snr}-{T60}-{example_idx}'] = metrics_tool.compute_all(target_clean_ref_1,
                                                                                              taylor_white_out_1)
                target_clean_ref_2, taylor_inter_out_2 = align_signal(target_clean_ref, taylor_inter_out)
                metrics[f'Taylor-inter-{snr}-{T60}-{example_idx}'] = metrics_tool.compute_all(target_clean_ref_2,
                                                                                              taylor_inter_out_2)

                if PLOT_AND_SAVE_FLAG and example_idx == EXAMPLE_IDX_TO_SAVE and snr == SNR_TO_SAVE and T60 == T60_TO_SAVE:
                    # White Noise
                    wavfile.write("output_folder_proj/taylor_white_out.wav", fs, taylor_white_out.astype(np.float32))
                    plot_time_freq_analysis(target_clean_ref / ORIGINAL_SIGNAL_FACTOR, ref_noisy_white,
                                            taylor_white_out, fs,
                                            f"(Taylor Output - White Noise - T60={T60}s - snr={snr}dB)",
                                            "Original", "Noisy", "Beamformer Out")

                    # Interferer
                    wavfile.write("output_folder_proj/taylor_interferer_out.wav", fs,
                                  taylor_inter_out.astype(np.float32))
                    plot_time_freq_analysis(target_clean_ref / ORIGINAL_SIGNAL_FACTOR, ref_noisy_inter,
                                            taylor_inter_out, fs,
                                            f"(Taylor Output - Interferer - T60={T60}s - snr={snr}dB)",
                                            "Original", "Noisy", "Beamformer Out")

                print("Deep Taylor Done.")

                all_metrics.append(metrics)

    parse_and_plot_results(all_metrics)


if __name__ == "__main__":
    taylor_main()
