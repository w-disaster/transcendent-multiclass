from src.feature_extraction.static.capa import CapaExtractor
from multiprocessing import Pool
from src.feature_extraction.config.config import config
from functools import partial

def build_capa_dataset(experiment, malware_dataset):
    # For singleton
    sha1s = malware_dataset.df_malware_family_fsd[["sha256", "family"]].to_numpy()

    capa_extractor = CapaExtractor()
    partial_extract = partial(capa_extractor.extract, experiment)
    
    with Pool(config.n_processes) as p:
        p.map(partial_extract, sha1s)
