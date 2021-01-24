from helper_classes import PYKE
from helper_classes import Parser
from helper_classes import DataAnalyser
from helper_classes import PPMI

import util as ut
import numpy as np


def main():
    try:
        print("Starting embeddings generation")
        random_state = 1
        np.random.seed(random_state)

        # DEFINE MODEL PARAMS
        K = 45
        num_of_dims = 100
        bound_on_iter = 30
        omega = 0.45557
        e_release = 0.0414

        kg_root = '../dbpedia/pyke_data'
        kg_path = kg_root + '/'

        print("Path to KG: ", kg_path)

        storage_path, experiment_folder = ut.create_experiment_folder()
        logger = create_logger(name='PYKE', p=storage_path)
        logger.info("Storage path: ", storage_path, "\texperiment folder: ", experiment_folder)
        parser = Parser(p_folder=storage_path, k=K)
        parser.set_logger(logger)
        logger.info("Setting similarity measure")
        parser.set_similarity_measure(PPMI)
        logger.info("Model init")
        model = PYKE(logger=logger)
        logger.info("Analyzer init")
        analyser = DataAnalyser(p_folder=storage_path, logger=logger)
        # For the illustration purpusoes lets only process first 5000 ntriples from each given file.
        # To reproduce  reported results => parser.pipeline_of_preprocessing(kg_path)
        holder = parser.pipeline_of_preprocessing(kg_path)

        vocab_size = len(holder)
        logger.info("Vocab size: ", vocab_size)
        embeddings = ut.randomly_initialize_embedding_space(vocab_size, num_of_dims)

        learned_embeddings = model.pipeline_of_learning_embeddings(e=embeddings,
                                                                   max_iteration=bound_on_iter,
                                                                   energy_release_at_epoch=e_release,
                                                                   holder=holder, omega=omega)

        logger.info("Writing to file.")
        learned_embeddings.to_csv(storage_path + '/PYKE_100_embd.csv')
        logger.info("Done!")
        # To use memory efficiently
        del holder
        del embeddings

    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
