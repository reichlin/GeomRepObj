To run the code launch on the terminal the command (code runnable on UNIX systems only):

python3 main.py --experiment <experiment_name> --move_bg <move_background> --rot <rotation> --model <model_name>

Dependencies are in the file requirements.txt

<experiment_name> can be one of: "2d" "3d" "3d_pov"
<move_background> can be one of: 0 1 (useful only in the experiment 2d)
<rotation>        can be one of: 0 1 (useful only in the experiment 2d)
<model_name>      can be one of: "haptic" "transporter" "vae" "end-to-end"

For both the experiments "3d", "3d_pov", please download the folder dataset (https://drive.google.com/file/d/1kb7zAurtcUhLSP2RBierNJmDq0450Iib/view?usp=sharing) and unzip it inside the folder 3d_experiments.

At the end of the training the trained model will be saved in the folder saved_models and the results will be saved as an .npy file in the folder saved_results,
results corresponds to the testing curves of the plots in the paper for the various experiments

--experiment "2d" --move_bg 0 --rot 0: corresponds to the sprite experiment of section 5.2 in the paper without the moving background
--experiment "2d" --move_bg 1 --rot 0: corresponds again to the sprite experiment of section 5.2 but with the blue squares moving background
--experiment "2d" --move_bg 0 --rot 1: corresponds again to the sprite experiment of section 5.2 but with the anisotropic object
--experiment "3d": corresponds to the soccer experiment of section 5.3. Pre-collected dataset is available in the "3d_experiment folder"
--experiment "3d_pov": corresponds to the soccer experiment of section 5.3 with the images from the cameras on top of the agent. Pre-collected dataset is available in the "3d_experiment folder"

All the above experiment can be run using a different model:
--model "haptic": corresponds to our proposed model described in the paper
--model "transporter": corresponds to the key-point extractor described in section 5.1 of the paper (first bullet point)
--model "vae": corresponds to the variational autoencoder described in section 5.1 of the paper (second bullet point)
--model "end-to-end": this model corresponds to a convolutional network trained with the RL loss only. This option can be chosen for the "rl" experiment only

the Control Task experiment described in section 5.4 is not runnable as it would require the executables for Unity, the code is however provided.
