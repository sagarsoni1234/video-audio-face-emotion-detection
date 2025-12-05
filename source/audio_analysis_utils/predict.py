import source.audio_analysis_utils.utils as utils
import source.config as config
import source.audio_analysis_utils.preprocess_data as data

import torch
import numpy as np
import os


def predict(input_file_name, model_save_path=config.AUDIO_MODEL_SAVE_PATH):
    audio_file = utils.find_filename_match(input_file_name, config.INPUT_FOLDER_PATH)
    if audio_file is None:
        raise Exception(f"Audio file not found: {input_file_name} in {config.INPUT_FOLDER_PATH}")
    
    audio_file_only = audio_file.split(config.INPUT_FOLDER_PATH)[1]
    print(f"audio_file: {audio_file_only}")

    best_hyperparameters = utils.load_dict_from_json(config.AUDIO_BEST_HP_JSON_SAVE_PATH)
    print(f"Best hyperparameters, {best_hyperparameters}")

    # Clean the audio file - handle any file extension
    base_name = os.path.splitext(audio_file_only)[0]  # Get filename without extension
    clean_audio_path = os.path.join(config.OUTPUT_FOLDER_PATH, base_name + '_clean.wav')
    data.clean_single(audio_file, save_path=clean_audio_path, print_flag=False)
    print("audio_file_cleaned")

    # Extract features
    extracted_mfcc = utils.extract_mfcc(
        clean_audio_path,
        N_FFT=best_hyperparameters['N_FFT'],
        NUM_MFCC=best_hyperparameters['NUM_MFCC'],
        HOP_LENGTH=best_hyperparameters['HOP_LENGTH']
    )
    print(f"extracted_mfcc: {extracted_mfcc.shape}")

    # Reshape to make sure it fit pytorch model
    extracted_mfcc = np.repeat(extracted_mfcc[np.newaxis, np.newaxis, :, :], 3, axis=1)
    print(f"Reshaped extracted_mfcc: {extracted_mfcc.shape}")

    # Convert to tensor
    extracted_mfcc = torch.from_numpy(extracted_mfcc).float().to(config.device)

    # Load the model (weights_only=False for PyTorch 2.6+ compatibility with custom classes)
    # map_location ensures models saved on GPU can be loaded on CPU
    model = torch.load(model_save_path, weights_only=False, map_location=config.device)
    model.to(config.device).eval()

    prediction = model(extracted_mfcc)
    prediction = torch.nn.functional.softmax(prediction, dim=1)
    prediction_numpy = prediction[0].cpu().detach().numpy()
    print(f"prediction: {prediction_numpy}")

    # Get the predicted label
    predicted_label = max(prediction_numpy)
    emotion = config.EMOTION_INDEX[prediction_numpy.tolist().index(predicted_label)]
    print(f"Predicted emotion: {emotion} {round(predicted_label, 2)}")

    ret_string = utils.get_softmax_probs_string(prediction_numpy, list(config.EMOTION_INDEX.values()))
    print(f"\n\n\nPrediction labels:\n{ret_string}")

    return emotion, round(predicted_label, 2), ret_string
