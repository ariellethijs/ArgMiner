import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import ast
import numpy as np
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from pipeline.arg_mining.issue_clustering import IssueClustering
from pipeline.arg_mining.triplet_identification import TripletsByStructureAndEmbeddingSimilarity
from pipeline.arg_mining.IRC_summary import IRC_summary


class ArgumentMiner():
    def __init__(self, casename):
        self.casename = casename
        self.tokenizer = AutoTokenizer.from_pretrained('zlucia/legalbert')
        self.has_main = False

        if not os.path.exists('./arg_mining/data/bert_labelled/' + casename + '.csv'):
            self.rhet_role_model = AutoModelForSequenceClassification.from_pretrained('zlucia/legalbert', num_labels=7)
            self.rhet_role_model_save_path = './arg_mining/model/legalbert_finetuned_rhet_roles.pth'
            self.rhet_role_model.load_state_dict(torch.load(self.rhet_role_model_save_path))
            self.rhet_role_model.eval()
            self.rhet_role_label_names = ["NONE", "TEXTUAL", "FACT", "PROCEEDINGS", "BACKGROUND", "FRAMING", "DISPOSAL"]
            self.label_case(casename)

        if not os.path.exists('./arg_mining/data/arg_role_labelled/' + casename + '.csv'):
            self.arg_role_model = AutoModelForSequenceClassification.from_pretrained('zlucia/legalbert', num_labels=4)
            self.arg_role_model_save_path = './arg_mining/model/legalbert_finetuned_arg_roles.pth'
            self.arg_role_model.load_state_dict(torch.load(self.arg_role_model_save_path))
            self.arg_role_model.eval()
            self.arg_role_label_names = ["NON-IRC", "ISSUE", "REASON", "CONCLUSION"]
            self.label_arg_components(casename)

    def mine_arguments(self, intro_para):
        # self.generate_sentence_bert_embeddings(self.casename)
        # issues_df = self.get_issue_df(self.casename)
        # print('Clustering issues')
        # clusterer = IssueClustering(issues_df, self.casename)
        # clusterer.process_issues()
        # print('Issues clustered\n')
        print('Determining main issue')
        main_issue = self.determine_issue_importance(self.casename)
        print('Main issue determined\n')
        print('Determining triplets')
        triplets_embed_struct = TripletsByStructureAndEmbeddingSimilarity(self.casename, main_issue, self.has_main)
        triples = triplets_embed_struct.process_judges()
        print('Triplets determined\n')
        print('Generating summary')
        summariser = IRC_summary(self.casename, triples, main_issue, self.has_main, intro_para)
        summary = summariser.generate_summary()
        print('Summary generated\n')
        return summary

    def predict_rhet_label(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.rhet_role_model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            return self.rhet_role_label_names[prediction]

    def label_case(self, casename):
        output_csv_path = './arg_mining/data/bert_labelled/' + casename + '.csv'
        if not os.path.exists(output_csv_path):
            print('Beginning Legal-BERT rhetorical classification')

            input_csv_path = './data/UKHL_corpus/' + casename + '.csv'
            df = pd.read_csv(input_csv_path)

            df['bert_role'] = df['text'].apply(
                lambda x: self.predict_rhet_label(x) if pd.notnull(x) and x.strip() != '' else None)

            df.to_csv(output_csv_path, index=False)
            print('Legal-BERT rhetorical classification complete')

    def predict_arg_label(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            # without embedding extraction:
            outputs = self.arg_role_model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            return self.arg_role_label_names[prediction]

            # with embedding extraction:
        #     outputs = self.arg_role_model(**inputs, output_hidden_states=True)
        #     hidden_states = outputs.hidden_states
        #     last_hidden_state = hidden_states[-1]
        #     embeddings = last_hidden_state.mean(dim=1)
        #     logits = outputs.logits
        #     prediction = torch.argmax(logits, dim=1).item()
        #     label = self.arg_role_label_names[prediction]
        # return label, embeddings

    def label_arg_components(self, casename):
        output_csv_path = './arg_mining/data/arg_role_labelled/' + casename + '.csv'
        if not os.path.exists(output_csv_path):
            print('Beginning Legal-BERT argument component classification')
            input_csv_path = './arg_mining/data/bert_labelled/' + casename + '.csv'

            df = pd.read_csv(input_csv_path)

            # without embeddings:
            df['arg_role'] = df['text'].apply(self.predict_arg_label)

            # with embeddings:
            # df[['arg_role', 'embeddings']] = df['text'].apply(
            #     lambda x: pd.Series(self.predict_arg_label(x)) if pd.notnull(x) and x.strip() != '' else pd.Series(
            #         [None, None]))
            # df['embeddings'] = df['embeddings'].apply(
            #     lambda x: x.squeeze().tolist() if pd.notnull(x) else None)

            df = self.check_arg_roles(df)
            df = self.validate_arg_paras(df)

            df.to_csv(output_csv_path, index=False)
            print('Legal-BERT argument component classification complete')

    def check_arg_roles(self, df):
        issue_rows = df[df['arg_role'] == 'ISSUE']

        for index, row in issue_rows.iterrows():
            if row['role'] != 'FRAMING' and row['bert_role'] != 'FRAMING':
                df.at[index, 'arg_role'] = 'NON-IRC'
        return df

    def validate_arg_paras(self, df):
        grouped = df.groupby('para_id')

        for para_id, group in grouped:
            if any(group['arg_role'] == 'REASON'):
                reason_count = (group['arg_role'] == 'REASON').sum()
                total_count = len(group)
                reason_per = (reason_count / total_count)

                if reason_count == 1:
                    non_reason_count = (group['arg_role'] == 'NON-IRC').sum()
                    only_arg_row = non_reason_count == (total_count - 1)
                else:
                    only_arg_row = False

                if reason_per < 0.15 or only_arg_row:
                    non_arg_roles = ['FACT', 'PROCEEDINGS', 'BACKGROUND']
                    bert_role_count = group['bert_role'].isin(non_arg_roles).sum()
                    bert_role_per = (bert_role_count / total_count)

                    if bert_role_per >= 0.80:
                        df.loc[group.index, 'arg_role'] = df.loc[group.index, 'arg_role'].replace('REASON', 'NON-IRC')
        return df

    def generate_sentence_bert_embeddings(self, casename):
        input = './arg_mining/data/arg_role_labelled/' + casename + '.csv'
        df = pd.read_csv(input)

        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        df['embeddings'] = df['text'].apply(lambda x: model.encode(x).tolist())
        df.to_csv(input, index=False)

    def get_issue_df(self, casename):
        entity_file_path = './summarydata-spacy/UKHL_' + casename + '.csv'
        entity_df = pd.read_csv(entity_file_path)
        labelled_arg_role_path = './arg_mining/data/arg_role_labelled/' + casename + '.csv'
        arg_df = pd.read_csv(labelled_arg_role_path)

        arg_df['entities'] = entity_df['entities']
        arg_df['embeddings'] = arg_df['embeddings'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).tolist()

        issues_df = arg_df[arg_df['arg_role'] == 'ISSUE']
        return issues_df

    def determine_issue_importance(self, casename):
        input_file_path = f'./arg_mining/data/issue_clustering/{casename}_issue_clusters.csv'
        df = pd.read_csv(input_file_path)

        importance_df = self.calculate_issue_importance_scores(df, casename)

        output_csv = './arg_mining/data/issue_ranking/' + casename + '_importance_scores.csv'
        importance_df.to_csv(output_csv, index=False)

        main_issue = self.find_main_issue(importance_df, casename)
        return main_issue

    def calculate_issue_importance_scores(self, df, casename):
        df['importance_score'] = 0
        main_cue_phrases, sub_cue_phrases = self.load_cue_phrases()

        # first issue mentioned by a judge?
        min_sentence_ids = df.groupby('judge')['sentence_id'].min()
        df['is_first'] = df.apply(lambda row: 1 if row['sentence_id'] == min_sentence_ids[row['judge']] else 0, axis=1)

        # mentioned by multiple judges?
        judge_counts = df.groupby('cluster')['judge'].nunique()
        df['mentions_multiple_judges'] = df.apply(lambda row: judge_counts[row['cluster']], axis=1)

        # mentioned multiple times?
        df['mention_count'] = df.groupby('cluster')['text'].transform('count')

        # no ordinal entities (often used when listing sub issues)
        df['has_ordinal_entities'] = df['entities'].apply(lambda x: 1 if '(ORDINAL)' in str(x) else 0)

        # judge saying is in majority
        df['judge_in_mj'] = df.apply(
            lambda row: 1 if any(judge in row['mj'] for judge in row['judge'].split('; ')) else 0, axis=1)

        # sentence was determined the representative sentence of its cluster
        df['is_representative'] = df.apply(
            lambda row: 1 if row['representative_sentence'] == row['text'] else 0, axis=1
        )

        # cue phrases for MAIN & SUB issues
        df['cue_phrase_main'] = df['text'].apply(
            lambda x: 1 if any(phrase in x.lower() for phrase in main_cue_phrases) else 0)
        df['cue_phrase_sub'] = df['text'].apply(
            lambda x: 1 if any(phrase in x.lower() for phrase in sub_cue_phrases) else 0)

        df['rhet_score'] = df.apply(lambda row: 1 if row['role'] == 'FRAMING' or row['bert_role'] == 'FRAMING' else 0,
                                    axis=1)

        df['normalised_is_first'] = self.normalise_column(df, 'is_first')
        df['normalised_mentions_multiple_judges'] = self.normalise_column(df, 'mentions_multiple_judges')
        df['normalised_mention_count'] = self.normalise_column(df, 'mention_count')
        df['normalised_has_ordinal_entities'] = self.normalise_column(df, 'has_ordinal_entities', invert=True)
        df['normalised_judge_in_mj'] = self.normalise_column(df, 'judge_in_mj')
        df['normalised_is_representative'] = self.normalise_column(df, 'is_representative')
        df['normalised_cue_phrase_main'] = self.normalise_column(df, 'cue_phrase_main')
        df['normalised_cue_phrase_sub'] = self.normalise_column(df, 'cue_phrase_sub', invert=True)
        df['normalised_role'] = self.normalise_column(df, 'rhet_score')

        weight_first = 0.2
        weight_judges = 0.1
        weight_mentions = 0.1
        weight_ordinal = 0.1
        weight_judge_in_mj = 0.2
        weight_representative = 0.05
        weight_cue_main = 0.1
        weight_cue_sub = 0.1
        weight_rhet_role = 0.05

        df['importance_score'] = (
                weight_first * df['normalised_is_first'] +
                weight_judges * df['normalised_mentions_multiple_judges'] +
                weight_mentions * df['normalised_mention_count'] +
                weight_ordinal * df['normalised_has_ordinal_entities'] +
                weight_judge_in_mj * df['normalised_judge_in_mj'] +
                weight_representative * df['normalised_is_representative'] +
                weight_cue_main * df['normalised_cue_phrase_main'] +
                weight_cue_sub * df['normalised_cue_phrase_sub'] +
                weight_rhet_role * df['normalised_role']
        )

        output_file = './arg_mining/data/issue_ranking/' + casename + '_importance_scores_with_calculations.csv'
        df.to_csv(output_file, index=False)

        return df[['text', 'cluster', 'importance_score']]

    def load_cue_phrases(self):
        file_path = './arg_mining/data/issue_type_cue_phrases.txt'
        main_cue_phrases = []
        sub_cue_phrases = []

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith("MAIN:"):
                    main_cue_phrases.append(line.replace("MAIN:", "").strip())
                elif line.startswith("SUB:"):
                    sub_cue_phrases.append(line.replace("SUB:", "").strip())

        return main_cue_phrases, sub_cue_phrases

    def normalise_column(self, df, column_name, invert=False):
        # normalise the scores for each column
        max_value = df[column_name].max()
        if max_value > 0:
            normalised = df[column_name] / max_value
            if invert:
                return 1 - normalised
            return normalised
        return df[column_name]

    def find_main_issue(self, df, casename):
        # rank issues based on importance
        top_score = df['importance_score'].max()
        if top_score >= 0.75:
            self.has_main = True
            top_issue = df.loc[df['importance_score'].idxmax()]
            cluster_id = top_issue['cluster']
            combined_df = pd.read_csv(f'./arg_mining/data/issue_clustering/{casename}_issue_clusters.csv')
            return combined_df[combined_df['cluster'] == cluster_id]['representative_sentence'].iloc[0]
        else:
            self.has_main = False
            return ''













