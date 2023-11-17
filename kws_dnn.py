import argparse
import os

import numpy as np

from utils.cuda import gpu_selection
from utils.deterministic import setup_deterministic_computation
import tensorflow as tf

from utils.forward import convert_dnn_model_to_reds, cross_entropy_loss, accuracy, train_model
from utils.importance_score import compute_accumulated_gradients_dnn, \
    compute_neurons_importance_score_dnn, compute_descending_filters_score_indexes_dnn, permute_neurons_dnn, \
    assign_pretrained_dense_layers
from utils.keyword_spotting import load_pre_trained_kws_model, compute_accuracy_test
from utils.keyword_spotting_data import get_audio_data
from utils.knapsack import knapsack_find_splits_dnn, initialize_nested_knapsack_solver_dnn
from utils.logs import setup_logging, log_print

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gurobi_home',
                        type=str,
                        default="",
                   
                        help="""\
                Gurobi Linux absolute path.
                """)

    parser.add_argument('--gurobi_license_file',
                        type=str,
                        default="",  
                        help="""\
                    Gurobi license absolute path.
                    """)

    parser.add_argument(
        '--data_url',
        type=str,
        default='http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
        help='Location of speech training data archive on the web.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='{}/datasets/speech_dataset/'.format(os.getcwd()),
        help="""\
            Where to download the speech training data to.
            """)
    parser.add_argument(
        '--background_volume',
        type=float,
        default=0.1,
        help="""\
            How loud the background noise should be, between 0 and 1.
            """)
    parser.add_argument(
        '--background_frequency',
        type=float,
        default=0.8,
        help="""\
            How many of the training samples have background noise mixed in.
            """)
    parser.add_argument(
        '--silence_percentage',
        type=float,
        default=10.0,
        help="""\
            How much of the training data should be silence.
            """)
    parser.add_argument(
        '--unknown_percentage',
        type=float,
        default=10.0,
        help="""\
            How much of the training data should be unknown words.
            """)
    parser.add_argument(
        '--time_shift_ms',
        type=float,
        default=100.0,
        help="""\
            Range to randomly shift the training audio by in time.
            """)
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a test set.')
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a validation set.')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs')
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs')
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=40,
        help='How long each spectrogram timeslice is')
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=40,
        help='How long each spectrogram timeslice is')
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=10,
        help='How many bins to use for the MFCC fingerprint')
    parser.add_argument(
        '--classes',
        type=int,
        default=12,
        help='How many classes the model needs to predict')
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=400,
        help='How often to evaluate the training results.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='How many items to train with at once')
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='/tmp/retrain_logs',
        help='Where to save summary logs for TensorBoard.')
    parser.add_argument(
        '--wanted_words',
        type=str,
        default='yes,no,up,down,left,right,on,off,stop,go',
        help='Words to use (others will be added to an unknown label)')
    parser.add_argument(
        '--architecture_name',
        type=str,
        default='dnn',
        help='What model architecture to use')
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='speech_dataset')

    parser.add_argument('--subnets_number', default=4, type=int, help='number of subnetworks to train')
    parser.add_argument(
        '--learning_rate',
        type=str,
        default='0.001,0.0001',
        help='How large a learning rate to use when training.')
    parser.add_argument(
        '--training_steps',
        type=str,
        default='15000,3000',
        help='How many training loops to run and learning rate schedule', )

    parser.add_argument('--cuda_device', default=12, type=int)
    parser.add_argument('--epochs', type=int, default=75, help='training epochs')
    parser.add_argument('--model_sizes', default='s', type=str,
                        help='model sizes')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--experimental_runs', default=3, type=int)
    parser.add_argument('--debug', default=False, action='store_true',
                        help='print intermediate activations and weights cuttings dimensions')
    parser.add_argument('--print', default=False, action='store_true',
                        help='print all the subnetworks accuracies')
    parser.add_argument('--plot', default=False, action='store_true',
                        help='plot the subnetworks finetuning and importance score')
    parser.add_argument('--minibatch_number', default=100, type=int)
    parser.add_argument('--save_path', default='{}/result/{}/KWS_Knapsack_alpha_{}_{}epochs_{}batch_{}subnetworks_{}',
                        type=str)
    parser.add_argument(
        '--constraints_percentages',
        type=str,
        default='0.25,0.5,0.75',
        help='Constraints percentages', )

    args, _ = parser.parse_known_args()
    setup_logging(args=args,
                  experiment_name="KWS_ARM_{}_importance_finetuning".format(args.architecture_name))

    setup_deterministic_computation(seed=args.seed)
    gpu_selection(gpu_number=args.cuda_device)

    os.environ[
        'GUROBI_HOME'] = args.gurobi_home
    os.environ['GRB_LICENSE_FILE'] = args.gurobi_license_file

    print("Gurobi settings:")
    print(os.getenv('GUROBI_HOME'))
    print(os.getenv('GRB_LICENSE_FILE'))

    layers_units = {
        's': 144,
        'm': 256,
        'l': 436
    }

    constraints_percentages = list(map(float, args.constraints_percentages.split(',')))

    for model_size in args.model_sizes.split(','):

        log_print("Loading model {} size {} minibatch number {} ".format('dnn', model_size, args.minibatch_number))

        average_final_subnetworks_accuracy, average_final_subnetworks_loss = [[] for _ in
                                                                              range(args.subnets_number)], [[] for _
                                                                                                            in
                                                                                                            range(
                                                                                                                args.subnets_number)]
        pretrained_model_accuracy = []

        subnetworks_macs_print = None

        for experimental_run in range(args.experimental_runs):

            training_steps_list = list(map(int, args.training_steps.split(',')))
            learning_rates_list = list(map(float, args.learning_rate.split(',')))
            lr_boundary_list = training_steps_list[:-1]
            lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=lr_boundary_list,
                                                                               values=learning_rates_list)

            pretrained_model, model_settings, model_size_info_dense = load_pre_trained_kws_model(
                args=args,
                model_name=args.architecture_name,
                model_size=model_size)

            train_data, val_data, test_data = get_audio_data(args=args, model_settings=model_settings)
            pretrained_model_test_accuracy = compute_accuracy_test(model=pretrained_model,
                                                                   model_settings=model_settings,
                                                                   test_data=test_data)

            pretrained_model_accuracy.append(pretrained_model_test_accuracy)

            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            gradients_accumulation = compute_accumulated_gradients_dnn(model=pretrained_model, train_data=train_data,
                                                                       loss_fn=loss_fn,
                                                                       args=args)

            importance_score_neurons = compute_neurons_importance_score_dnn(model=pretrained_model,
                                                                            gradients_accumulation=gradients_accumulation)

            descending_importance_score_indexes, descending_importance_score_scores = compute_descending_filters_score_indexes_dnn(
                model=pretrained_model,
                importance_score_neurons=importance_score_neurons)

            permuted_dense_neurons, permuted_dense_bias = permute_neurons_dnn(
                model=pretrained_model,
                neurons_descending_ranking=descending_importance_score_indexes)

            assign_pretrained_dense_layers(model=pretrained_model,
                                           permuted_dense_weights=permuted_dense_neurons,
                                           permuted_dense_bias=permuted_dense_bias,
                                           trainable_assigned_dense=True)

            reds_pretrained_model = convert_dnn_model_to_reds(pretrained_model=pretrained_model, train_ds=train_data,
                                                              args=args,
                                                              model_settings=model_settings,
                                                              trainable_parameters=True,
                                                              hidden_units=layers_units[model_size])

            layers_filters_macs, layers_filters_byte = reds_pretrained_model.compute_lookup_table(
                train_data=train_data)

            importance_list, macs_list, memory_list, macs_targets, memory_targets = initialize_nested_knapsack_solver_dnn(
                layers_filters_macs=layers_filters_macs, layers_filters_byte=layers_filters_byte,
                subnetworks_number=args.subnets_number,
                descending_importance_score_scores=descending_importance_score_scores,
                constraints_percentages=constraints_percentages)

            subnetworks_neurons_indexes, subnetworks_macs = knapsack_find_splits_dnn(args=args,
                                                                                     layers_filter_macs=layers_filters_macs,
                                                                                     importance_list=descending_importance_score_scores,
                                                                                     macs_targets=macs_targets,
                                                                                     macs_list=macs_list,
                                                                                     memory_list=memory_list,
                                                                                     memory_targets=memory_targets,
                                                                                     bottom_up=True)

            if subnetworks_macs_print is None:
                subnetworks_macs_print = subnetworks_macs

            reds_pretrained_model.set_subnetwork_indexes(subnetworks_neurons_indexes=subnetworks_neurons_indexes)

            optimizer = tf.keras.optimizers.experimental.Adam(learning_rate=lr_schedule)
            optimizer.build(var_list=reds_pretrained_model.trainable_variables)

            final_subnetworks_accuracy = train_model(model=reds_pretrained_model, train_data=train_data,
                                                     test_data=test_data,
                                                     val_data=val_data, debug=args.debug,
                                                     plot=args.plot,
                                                     loss=cross_entropy_loss, acc=accuracy,
                                                     message_initial_accuracies="Initial accuracy REDS model WITH filters permutation WITH finetuned classification head",
                                                     architecture_name=args.architecture_name + f"_{model_size}",
                                                     importance_score=True,
                                                     message=f"REDS finetuning {reds_pretrained_model.get_model_name()} KWS pretrained model WITH filters permutation",
                                                     optimizer=optimizer, epochs=args.epochs,
                                                     subnetworks_number=args.subnets_number, args=args,
                                                     subnetworks_macs=subnetworks_macs)

            for subnetwork_index in range(args.subnets_number):
                print(
                    f"Experimental run: {experimental_run} Subnetwork {subnetworks_macs[subnetwork_index]} MACs test accuracy: {100 * np.array(final_subnetworks_accuracy[subnetwork_index]).mean()}%")

            [average_final_subnetworks_accuracy[subnetwork_number].append(
                100 * final_subnetworks_accuracy[subnetwork_number][0])
                for subnetwork_number in range(args.subnets_number)]

            reds_pretrained_model = convert_dnn_model_to_reds(pretrained_model=pretrained_model, train_ds=train_data,
                                                              args=args,
                                                              model_settings=model_settings,
                                                              trainable_parameters=True,
                                                              hidden_units=layers_units[model_size])

            reds_pretrained_model.set_subnetwork_indexes(subnetworks_neurons_indexes=subnetworks_neurons_indexes)

        for subnetwork_number in range(args.subnets_number):
            log_print(
                f"subnetworks MACS: {subnetworks_macs_print[subnetwork_number]} test accuracy mean: {np.array(average_final_subnetworks_accuracy[subnetwork_number]).mean():.3f}% test accuracy std: {np.array(average_final_subnetworks_accuracy[subnetwork_number]).std():.3f}")

        log_print(
            f"pretrained model average test accuracy: {np.array(pretrained_model_accuracy).mean():.3f}% std: {np.array(pretrained_model_accuracy).std():.3f}")

