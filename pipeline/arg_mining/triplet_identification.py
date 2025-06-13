import ast
import json

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def generate_df(casename, main_issue):
    entity_file_path = f'./summarydata-spacy/UKHL_{casename}.csv'
    labelled_arg_role_path = f'./arg_mining/data/arg_role_labelled/{casename}.csv'
    entity_df = pd.read_csv(entity_file_path)
    df = pd.read_csv(labelled_arg_role_path)
    df['entities'] = entity_df['entities']

    cluster_file_path = f'./arg_mining/data/issue_clustering/{casename}_issue_clusters.csv'
    cluster_df = pd.read_csv(cluster_file_path)
    cluster_df = cluster_df[['text', 'cluster', 'representative_sentence']]
    df = pd.merge(df, cluster_df, on='text', how='left')

    df = mark_main_issues(df, main_issue)
    df = mark_main_conclusions(df)

    df['issue_index'] = -1

    # issue_index mapping for all issues
    issues = df[df['arg_role'] == 'ISSUE']['text'].unique()
    issue_to_index = {issue: idx for idx, issue in enumerate(issues) if issue != main_issue}
    issue_to_index[main_issue] = 0  # main_issue is indexed as 0
    df['issue_index'] = df['text'].map(issue_to_index).fillna(-1)

    df.loc[(df['arg_role'] == 'ISSUE') & (df['main_issue'] == 1), 'issue_index'] = 0

    df['embeddings'] = df['embeddings'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).tolist()
    return df


def mark_main_issues(df, main_issue):
    df['main_issue'] = 0

    condition1 = (df['representative_sentence'] == main_issue)
    condition2 = (df['text'] == main_issue)
    condition3 = df['text'].apply(
        lambda x: x in main_issue and len(x) >= 20 if pd.notna(x) and pd.notna(main_issue) else False
    )

    condition = condition1 | condition2 | condition3

    df.loc[condition, 'main_issue'] = 1
    return df


def mark_main_conclusions(df):
    df['main_conclusion'] = 0

    for judge, group in df.groupby('judge'):
        condition = (group['role'] == 'DISPOSAL') & (group['arg_role'] == 'CONCLUSION')
        matching_rows = group[condition]

        if not matching_rows.empty:
            matching_indices = matching_rows.index
            for idx in sorted(matching_indices, reverse=True):
                df.loc[idx, 'main_conclusion'] = 1
                if idx - 1 in matching_indices:
                    continue
                else:
                    break

    return df


class TripletsByStructureAndEmbeddingSimilarity:
    def __init__(self, casename, main_issue, has_main):
        self.casename = casename
        self.output_csv = f'./arg_mining/data/IRC_triples/{casename}_structure_and_embeddings.csv'
        df = generate_df(casename, main_issue)
        self.df = df
        self.df = self.df[self.df['judge'].notna() & (self.df['judge'] != 'None')]

        self.has_main = has_main
        self.main_issue = main_issue
        main_issue_row = self.df[self.df['text'] == self.main_issue]
        if not main_issue_row.empty:
            self.main_issue_id = main_issue_row['sentence_id'].iloc[0]
        else:
            self.main_issue_id = 0

        self.results = {}

    def process_judges(self):
        for judge in self.df['judge'].unique():
            judge_df = self.df[self.df['judge'] == judge]
            issue_texts = judge_df[judge_df['arg_role'] == 'ISSUE']
            reason_texts = judge_df[judge_df['arg_role'] == 'REASON']
            conclusion_texts = judge_df[judge_df['arg_role'] == 'CONCLUSION']

            if not issue_texts.empty:
                issue_dicts = self.assign_reasons_to_issues(issue_texts, reason_texts)
                issue_dicts = self.assign_conclusions_to_issues(issue_texts, conclusion_texts, issue_dicts)

                main_issue_texts = judge_df[judge_df['main_issue'] == 1]['text'].tolist()
                if not main_issue_texts and self.has_main:
                    main_issue_texts = [self.main_issue]
                elif not main_issue_texts and not self.has_main:
                    main_issue_texts = ''

                self.results[judge] = {
                    'main_issue': {
                        'issue': main_issue_texts,
                        'reasons': [
                            reason for issue_id, issue_dict in issue_dicts.items()
                            if any(self.main_issue in issue for issue in issue_dict['issue'])
                            for reason in issue_dict['reasons']
                        ] if self.has_main else [],
                        'conclusions': [
                            conclusion for issue_id, issue_dict in issue_dicts.items()
                            if any(self.main_issue in issue for issue in issue_dict['issue'])
                            for conclusion in issue_dict['conclusions']
                        ] if self.has_main else []
                    },
                    'sub_issues': [
                        {'issue': issue_dict['issue'], 'reasons': issue_dict['reasons'],
                         'conclusions': issue_dict['conclusions']}
                        for issue_id, issue_dict in issue_dicts.items()
                        if not issue_dict['issue'] == self.main_issue
                    ]
                }

                main_conclusion_texts = judge_df[judge_df['main_conclusion'] == 1]['text'].tolist()
                if main_conclusion_texts:
                    self.results[judge]['main_issue']['conclusions'].extend(main_conclusion_texts)
            else:
                # no issues, use main issue for all
                self.results[judge] = {
                    'main_issue': {
                        'issue': [self.main_issue],
                        'reasons': list(reason_texts['text']),
                        'conclusions': list(conclusion_texts['text'])
                    }
                }

        self.df.to_csv(self.output_csv)
        # selector = KeyIssueSelector(self.casename, )
        ranker = ReasonsRanker(self.df, self.results, self.casename)
        self.results = ranker.rank_reasons()
        return self.results

    def calculate_similarity(self, embedding1, embedding2):
        return cosine_similarity([embedding1], [embedding2])[0][0]

    def extract_law_entities(self, entities_str):
        if pd.isna(entities_str):
            return set()
        else:
            entities = entities_str.split(';')
            return {e.strip() for e in entities if 'LAW' in e}

    def calculate_entity_matching_score(self, issue_entities, reason_entities):
        common_entities = len(issue_entities.intersection(reason_entities))
        return common_entities / (len(issue_entities) + 1) if issue_entities else 0

    def assign_reasons_to_issues(self, issue_texts, reason_texts):
        issue_dicts = {row['sentence_id']: {'issue': row['text'], 'reasons': [], 'conclusions': []}
                       for _, row in issue_texts.iterrows()}

        for _, reason_row in reason_texts.iterrows():
            reason_id = reason_row['sentence_id']
            reason_embedding = np.array(reason_row['embeddings'])
            reason_entities = self.extract_law_entities(reason_row['entities'])
            scores = []

            for _, issue_row in issue_texts.iterrows():
                issue_id = issue_row['sentence_id']
                issue_embedding = np.array(issue_row['embeddings'])
                issue_entities = self.extract_law_entities(issue_row['entities'])

                similarity = self.calculate_similarity(reason_embedding, issue_embedding)
                avg_entity_match = self.calculate_entity_matching_score(issue_entities, reason_entities)

                proximity_score = 0
                if issue_id < reason_id and not issue_texts[
                        (issue_texts['sentence_id'] > issue_row['sentence_id']) &
                        (issue_texts['sentence_id'] < reason_row['sentence_id'])
                    ].shape[0]:
                    proximity_score = 1  # close w/o other issues in between

                # exclude proximity from main_issue calc as does not follow traditional structure
                if issue_row.get('main_issue') == 1 and self.has_main:
                    similarity_weight = 0.7
                    entity_weight = 0.3
                    final_score = (similarity_weight * similarity + entity_weight * avg_entity_match)
                else:
                    similarity_weight = 0.2
                    proximity_weight = 0.6
                    entity_weight = 0.2
                    final_score = (similarity_weight * similarity + proximity_weight * proximity_score +
                                   entity_weight * avg_entity_match)

                scores.append((issue_id, final_score))

            # match which issue the reason fits best
            best_issue_id = max(scores, key=lambda x: x[1])[0] if scores else None

            if best_issue_id:
                issue_dicts[best_issue_id]['reasons'].append(reason_row['text'])
                issue_index = self.df.loc[self.df['sentence_id'] == issue_id, 'issue_index'].values[0]
                self.df.loc[self.df['sentence_id'] == reason_id, 'issue_index'] = issue_index

        return issue_dicts

    def assign_conclusions_to_issues(self, issue_texts, conclusion_texts, issue_dicts):
        for _, conclusion_row in conclusion_texts.iterrows():
            conclusion_id = conclusion_row['sentence_id']

            if conclusion_row.get('main_conclusion') == 1:
                # skip main conclusions as they automatically be added to main_issue
                self.df.loc[self.df['sentence_id'] == conclusion_id, 'issue_index'] = 0
                continue

            conclusion_embedding = np.array(conclusion_row['embeddings'])
            conclusion_entities = self.extract_law_entities(conclusion_row['entities'])
            scores = []

            for _, issue_row in issue_texts.iterrows():
                # get sentence_id, embeddings, and entities ready
                issue_id = issue_row['sentence_id']
                issue_embedding = np.array(issue_row['embeddings'])
                issue_entities = self.extract_law_entities(issue_row['entities'])

                similarity = self.calculate_similarity(conclusion_embedding, issue_embedding)

                avg_entity_match = self.calculate_entity_matching_score(issue_entities, conclusion_entities)

                proximity_score = 0
                if issue_id < conclusion_id and not issue_texts[
                        (issue_texts['sentence_id'] > issue_row['sentence_id']) &
                        (issue_texts['sentence_id'] < conclusion_row['sentence_id'])
                    ].shape[0]:
                    proximity_score = 1  # close enough without other issues in between

                if issue_row.get('main_issue') == 1:
                    similarity_weight = 0.8 # non-main conclusions should be VERY similar to attribute to main_issue
                    entity_weight = 0.2
                    final_score = (similarity_weight * similarity + entity_weight * avg_entity_match)
                else:
                    similarity_weight = 0.2
                    proximity_weight = 0.6 # Higher proximity weight for conclusions of sub-issues
                    entity_weight = 0.2
                    final_score = (similarity_weight * similarity + proximity_weight * proximity_score +
                                   entity_weight * avg_entity_match)

                scores.append((issue_id, final_score))

            # select best issue
            best_issue_id = max(scores, key=lambda x: x[1])[0] if scores else None
            issue_dicts[best_issue_id]['conclusions'].append(conclusion_row['text'])

            issue_index = self.df.loc[self.df['sentence_id'] == issue_id, 'issue_index'].values[0]
            self.df.loc[self.df['sentence_id'] == conclusion_id, 'issue_index'] = issue_index

        return issue_dicts


class ReasonsRanker:
    def __init__(self, df, results, casename):
        self.df = df
        self.results = results
        self.casename = casename
        self.ranked_results = {}

    def calculate_similarity(self, emb1, emb2):
        return cosine_similarity([emb1], [emb2])[0][0]


    def extract_law_entities(self, entities_str):
        if pd.isna(entities_str):
            return set()
        else:
            return {entity.strip() for entity in entities_str.split(';') if 'LAW' in entity}

    def rank_reasons(self):
        for judge, judge_data in self.results.items():
            self.ranked_results[judge] = {}
            for issue_type, section_data in judge_data.items():
                if issue_type == 'main_issue':
                    self.ranked_results[judge][issue_type] = self.rank_section_reasons(section_data)
                elif issue_type == 'sub_issues':
                    self.ranked_results[judge][issue_type] = []
                    for sub_issue_data in section_data:
                        sub_issue_ranked = self.rank_section_reasons(sub_issue_data)
                        self.ranked_results[judge][issue_type].append(sub_issue_ranked)

        self.results_to_json()
        return self.ranked_results

    def rank_section_reasons(self, section_data):
        issues = section_data['issue']
        reasons = section_data['reasons']
        conclusions = section_data['conclusions']

        issue_embeddings = []
        issue_ids = []
        issue_entities = []

        if isinstance(issues, (list, tuple)):
            for issue_text in issues:
                issue_row = self.df[(self.df['text'] == issue_text) & (self.df['arg_role'] == 'ISSUE')]
                if not issue_row.empty:
                    issue_embeddings.append(issue_row['embeddings'].values[0])
                    issue_ids.append(issue_row['sentence_id'].values[0])
                    issue_entities.append(self.extract_law_entities(issue_row['entities'].values[0]))
        else:
            issue_row = self.df[(self.df['text'] == issues) & (self.df['arg_role'] == 'ISSUE')]
            if not issue_row.empty:
                issue_embeddings.append(issue_row['embeddings'].values[0])
                issue_ids.append(issue_row['sentence_id'].values[0])
                issue_entities.append(self.extract_law_entities(issue_row['entities'].values[0]))

        law_entities_present = any(issue_entities)

        ranked_reasons = []
        for reason in reasons:
            reason_row = self.df[(self.df['text'] == reason) & (self.df['arg_role'] == 'REASON')]
            if reason_row.empty:
                continue
            reason_emb = reason_row['embeddings'].values[0]
            reason_id = reason_row['sentence_id'].values[0]
            reason_entities = self.extract_law_entities(reason_row['entities'].values[0])
            reason_role = reason_row['bert_role'].values[0]

            similarity_scores = [self.calculate_similarity(issue_emb, reason_emb) for issue_emb in
                                 issue_embeddings]
            avg_similarity = np.mean(similarity_scores) if similarity_scores else 0

            proximity_scores = []
            for conclusion in conclusions:
                conclusion_row = self.df[(self.df['text'] == conclusion) & (self.df['arg_role'] == 'CONCLUSION')]
                if not conclusion_row.empty:
                    conclusion_id = conclusion_row['sentence_id'].values[0]
                    if conclusion_id and reason_id:
                        proximity_scores.append(1 / (abs(conclusion_id - reason_id) + 1))

            avg_proximity = np.mean(proximity_scores) if proximity_scores else 0

            entity_matching_score = 0
            for issue_entities_set in issue_entities:
                common_entities = len(issue_entities_set.intersection(reason_entities))
                entity_matching_score += common_entities / (
                        len(issue_entities_set) + 1)
            avg_entity_match = entity_matching_score / len(issue_entities) if issue_entities else 0

            if law_entities_present:
                similarity_weight = 0.5
                proximity_weight = 0.3
                entity_weight = 0.2
                final_score = (similarity_weight * avg_similarity + proximity_weight * avg_proximity +
                               entity_weight * avg_entity_match)
            else:
                similarity_weight = 0.7
                proximity_weight = 0.3
                final_score = (similarity_weight * avg_similarity + proximity_weight * avg_proximity)

            ranked_reasons.append({
                'reason_text': reason,
                'avg_similarity': avg_similarity,
                'avg_proximity': avg_proximity,
                'avg_entity_match': avg_entity_match,
                'final_score': final_score,
                'sentence_id': reason_id,
                'role': reason_role
            })

        ranked_reasons = sorted(ranked_reasons, key=lambda x: x['final_score'], reverse=True)

        # for reason in ranked_reasons:
        #     text = reason['reason_text']
        #     score = reason['final_score']
        #     print(f'{text}, has score of {score}\n\n')

        # # selection w/ Hachey and Grover's distribution of rhetorical roles
        selected_reasons = self.select_HG_distribution(ranked_reasons, len(ranked_reasons))

        # Select 30% of the reasons based on how complex the issue is (how many reasons attributed to it)
        # num_to_select = int(0.3 * len(ranked_reasons))
        # selected_reasons = ranked_reasons[:num_to_select]

        # selection w/top ranked background, framing, and disposal
        # selected_reasons = self.select_distribution(ranked_reasons, len(ranked_reasons))

        selected_reasons = sorted(selected_reasons, key=lambda x: x['sentence_id'])

        issue_ranked = {
            'issue': section_data['issue'],
            'reasons': [reason['reason_text'] for reason in selected_reasons],
            'conclusions': section_data['conclusions']
        }

        # # Single REASON implementation
        # selected_reason = ranked_reasons[:1]
        # print(selected_reason)
        # issue_ranked = {
        #     'issue': section_data['issue'],
        #     'reasons': selected_reason,
        #     'conclusions': section_data['conclusions']
        # }

        return issue_ranked

    def select_top_reasons(self, ranked_reasons, role, top_n):
        # print(f'{role}: {top_n}')
        filtered_reasons = [r for r in ranked_reasons if r['role'] == role]
        # print(f'{filtered_reasons[:top_n]}\n\n')
        return filtered_reasons[:top_n]

    def select_distribution(self, ranked_reasons, total_reasons):
        num_to_select = int(0.3 * total_reasons)  # 30% of total reasons
        print(f'Selecting {num_to_select} reasons of a total of {total_reasons}')
        background_dist = 33
        framing_dist = 34
        disposal_dist = 33
        num_background = int((background_dist / 100) * num_to_select)
        num_framing = int((framing_dist / 100) * num_to_select)
        num_disposal = int((disposal_dist / 100) * num_to_select)
        print(f'Selecting {num_background} from BACKGROUND & {num_framing} from FRAMING & {num_disposal} from DISPOSAL\n')

        selected_reasons = (
                self.select_top_reasons(ranked_reasons, 'BACKGROUND', num_background) +
                self.select_top_reasons(ranked_reasons, 'FRAMING', num_framing) +
                self.select_top_reasons(ranked_reasons, 'DISPOSAL', num_disposal)
        )

        return selected_reasons

    def select_HG_distribution(self, ranked_reasons, total_reasons):
        if total_reasons < 4:
            # always select at least:
            num_background = 0
            num_framing = 2
            num_proceedings = 0
            num_fact = 0
            num_disposal = 1
        else:
            background_dist = 10.2
            framing_dist = 30.0
            proceedings_dist = 18.4
            fact_dist = 10.3
            disposal_dist = 31.1

            print(total_reasons)

            num_to_select = int(0.3 * total_reasons)  # 30% of total reasons
            print(f'Selecting {num_to_select} reasons of a total of {total_reasons}')
            num_background = int((background_dist / 100) * num_to_select)
            num_framing = int((framing_dist / 100) * num_to_select)
            num_proceedings = int((proceedings_dist / 100) * num_to_select)
            num_fact = int((fact_dist / 100) * num_to_select)
            num_disposal = int((disposal_dist / 100) * num_to_select)

            # print(f'Selecting {num_to_select} reasons of {total_reasons} total')

        selected_reasons = (
                self.select_top_reasons(ranked_reasons, 'FRAMING', num_framing) +
                self.select_top_reasons(ranked_reasons, 'BACKGROUND', num_background) +
                self.select_top_reasons(ranked_reasons, 'PROCEEDINGS', num_proceedings) +
                self.select_top_reasons(ranked_reasons, 'FACT', num_fact) +
                self.select_top_reasons(ranked_reasons, 'DISPOSAL', num_disposal)
        )

        return selected_reasons

    def results_to_json(self):
        filepath = f'./arg_mining/data/IRC_triples/ranked_triples/{self.casename}.json'
        with open(filepath, 'w') as file:
            json.dump(self.ranked_results, file, indent=4)









