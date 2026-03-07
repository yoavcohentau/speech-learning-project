# Speech Learning Project - Taylor Beamformer

# Project Structure

* **`\DataSetGeneration`** - Scripts we wrote to generate the datasets and compare the different algorithms.
* **`\DeepTaylorBeamformer`** - Based on the original TaylorBeamformer code. Contains the network models, training loop, and inference scripts.
* **`\utils`** - Helper functions for the classic baseline algorithms, plus general tools for running the experiments and plotting results.

---

### Main Files & How to Run

**`proj_main.py`** - The main script to run the test simulation.

**Notes:**
- Change `DATA_SET_PATH` to your local LibriSpeech folder.
- If you want to save audio files or plot graphs, change the consts at the top of the script (PLOT_AND_SAVE_FLAG, EXAMPLE_IDX_TO_SAVE, SNR_TO_SAVE, T60_TO_SAVE). Please make sure a folder named 'output_folder_proj' exists in the project directory.
- To run the simulation with a mismatched microphone array layout, change the `mic_positions` variable inside the `generate_room_impulse_responses` function (found in `utils/speech_generator_utils.py`). Just swap the linear array code with the fixed array (the alternative code is already there, commented out).

**`file_name_list.yaml`** - The list of audio files used in the simulation.

**`final_project_207319849_208852715.ipynb`** - Our training notebook (with Google Colab). You can also find it in our Google Drive with the datasets we generated:
[Google Drive Link](https://drive.google.com/drive/folders/1VuGfjQJLzuBG-zXftti9K-8vPgZgmPu8?usp=drive_link)

---

**Based on the paper:**
*TaylorBeamformer: Learning All-Neural Beamformer for Multi-Channel Speech Enhancement from Taylor’s Approximation Theory*