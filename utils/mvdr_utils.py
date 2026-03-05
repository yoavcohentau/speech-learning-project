import librosa
import numpy as np
from scipy.linalg import eigh
from scipy.linalg import pinv


def estimate_cov_matrix(stft_mat):
    X = np.asarray(stft_mat, dtype=np.complex128)
    t = X.shape[2]
    R = np.einsum('mft,nft->fmn', X, X.conj()) / t

    return R


def estimate_rtf_using_gevd(noisy_stft, noise_cov, ref_mic=2):
    R_y = estimate_cov_matrix(noisy_stft)

    # find the rtf per frequency
    n_freqs, n_channels = R_y.shape[0], R_y.shape[1]
    rtf = np.zeros((n_channels, n_freqs), dtype=np.complex128)
    for f in range(n_freqs):
        R_n = noise_cov[f] + 1e-3 * np.eye(n_channels)
        _, vecs = eigh(R_y[f], R_n)  # calc GEVD
        v = vecs[:, -1]
        h = R_n @ v  # calc RTF
        rtf[:, f] = h / (h[ref_mic] + 1e-12)  # normalize by ref mic
    # rtf = rtf + 0.3*np.random.randn(*rtf.shape)

    return rtf


def _compute_mvdr_weights(noise_cov, rtf):
    n_freqs, n_mics, _ = noise_cov.shape
    weights = np.zeros((n_mics, n_freqs), dtype=np.complex128)

    for f in range(n_freqs):
        R_inv = pinv(noise_cov[f])
        h = rtf[:, f]

        numerator = R_inv @ h  # MVDR Numerator: R^-1 * h
        denominator = (h.conj().T @ numerator).real  # MVDR Denominator: h^H * R^-1 * h

        if denominator < 1e-15:
            weights[:, f] = 0
        else:
            weights[:, f] = numerator / denominator

    return weights


def apply_mvdr(mic_signals, noise, win_length=800, hop_length=48,
               is_return_stft=False, use_known_noise=False):
    mic_signals_stft = librosa.stft(mic_signals, n_fft=win_length, hop_length=hop_length, win_length=win_length)
    if use_known_noise:
        noise_stft = librosa.stft(noise, n_fft=win_length, hop_length=hop_length, win_length=win_length)
    else:
        fs = 16000
        noise_duration_samples = int(0.02 * fs)
        noise_ref = mic_signals[:, :noise_duration_samples]
        noise_stft = librosa.stft(noise_ref, n_fft=win_length, hop_length=hop_length, win_length=win_length)

    noise_cov = estimate_cov_matrix(noise_stft)
    # noise_cov = noise_cov + 0.1 * np.random.randn(*noise_cov.shape)

    rtf = estimate_rtf_using_gevd(mic_signals_stft, noise_cov)

    w = _compute_mvdr_weights(noise_cov, rtf)

    X = np.asarray(mic_signals_stft, dtype=np.complex128)
    w = np.asarray(w, dtype=np.complex128)
    M, F, N = X.shape
    out_stft = np.zeros((F, N), dtype=np.complex128)
    for f in range(F):  # apply filter
        out_stft[f, :] = w[:, f].conj().T @ X[:, f, :]

    out_mvdr = librosa.istft(
        out_stft, hop_length=hop_length, win_length=win_length, n_fft=win_length, length=np.shape(mic_signals)[1])

    if is_return_stft:
        return out_stft, (hop_length, win_length, win_length, np.shape(mic_signals)[1])

    return np.real(out_mvdr)
