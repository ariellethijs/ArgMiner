import HTMLTextExtractor
import asmo_pipeline
import prepare_labelling
import labelling
import featureExtractor
import ml
import summary
from pipeline.arg_mining import arg_mining


class IRC_pipeline:
    def begin(self):
        print('Enter a link to evaluate a case:')
        url = input()
        filepath = HTMLTextExtractor.HTMLTextExtractor(url).extract_text()
        casename = filepath.split("/")[-1].split(".")[0]
        asmo_pipeline.asmo(filepath)
        prepare_labelling.prepare_labelling(filepath)
        labelling.labelling(casename)
        featureExtractor.featureExtractor(casename)
        ml.ml(casename, True)
        miner = arg_mining.ArgumentMiner(casename)
        SUMO = summary.summary(casename, False)
        intro_para = SUMO.generate_preIRC_para(casename)
        IRCsummary = miner.mine_arguments(intro_para)
        return IRCsummary


pipeline = IRC_pipeline()
pipeline.begin()