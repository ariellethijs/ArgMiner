# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.cluster import AgglomerativeClustering
# from sentence_transformers import SentenceTransformer
#
# class IssueClustering:
#     def __init__(self, issues_df, casename):
#         self.output = f'./arg_mining/data/issue_clustering/{casename}_issue_clusters.csv'
#         self.casename = casename
#         self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#         self.issues_df = issues_df
#         self.entity_similarity_threshold = 0.85
#         self.final_similarity_threshold = 0.75
#         self.semantic_weight = 0.7
#         self.entity_weight = 0.3
#         self.grouped_issues = []
#
#     def process_issues(self):
#         valid_judges = self.issues_df['judge'].apply(lambda x: x not in [None, 'None'])
#         self.issues_df = self.issues_df[valid_judges]
#         self.issues_df['entities'] = self.issues_df['entities'].apply(self.extract_entities)
#
#         # Compute embeddings
#         self.issues_df['embeddings'] = self.issues_df['text'].apply(lambda x: self.model.encode(x))
#
#         similarity_matrix = self.compute_similarity_matrix()
#         clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average',
#                                              distance_threshold=1 - self.final_similarity_threshold)
#         cluster_labels = clustering.fit_predict(1 - similarity_matrix)
#
#         self.issues_df['cluster'] = cluster_labels
#         self.store_representative_sentence()
#         self.issues_df.to_csv(self.output, index=False)
#
#     def compute_similarity_matrix(self):
#         num_issues = len(self.issues_df)
#         similarity_matrix = np.zeros((num_issues, num_issues))
#
#         for i in range(num_issues):
#             for j in range(i + 1, num_issues):
#                 sim_score = self.compute_combined_similarity(self.issues_df.iloc[i], self.issues_df.iloc[j])
#                 similarity_matrix[i, j] = sim_score
#                 similarity_matrix[j, i] = sim_score
#
#         return similarity_matrix
#
#     def compute_combined_similarity(self, issue1, issue2):
#         text1 = issue1['text']
#         text2 = issue2['text']
#
#         semantic_similarity = cosine_similarity([issue1['embeddings']], [issue2['embeddings']])[0][0]
#         print(f"Comparing:\nText 1: {text1}\nText 2: {text2}")
#         print(f"Semantic Similarity: {semantic_similarity:.2f}")
#
#         entity_similarity = 0
#         if issue1['entities'] and issue2['entities']:
#             entity_similarity = self.compute_entity_scores(issue1['entities'], issue2['entities'])
#             print(f"Entity Similarity: {entity_similarity:.2f}")
#
#         if issue1['entities']:
#             final_score = (self.semantic_weight * semantic_similarity +
#                            self.entity_weight * entity_similarity)
#         else:
#             final_score = semantic_similarity
#         print(f"Combined Similarity Score: {final_score:.2f}")
#
#         # Print the verdict
#         if final_score >= self.final_similarity_threshold:
#             print("Similarity Verdict: Similar")
#         else:
#             print("Similarity Verdict: Not Similar")
#         print("-" * 50)
#
#         return final_score
#
#     def compute_entity_scores(self, entities1, entities2):
#         highest_similarity = float('-inf')
#         print(f"Computing entity scores between entities:\nEntities 1: {entities1}\nEntities 2: {entities2}")
#
#         for ent1 in entities1:
#             for ent2 in entities2:
#                 similarity = self.compute_entity_similarity(ent1, ent2)
#                 print(f"Entity Comparison: '{ent1}' vs '{ent2}'")
#                 print(f"Entity Similarity Score: {similarity:.2f}")
#                 highest_similarity = max(highest_similarity, similarity)
#
#         return highest_similarity
#
#     def compute_entity_similarity(self, ent1, ent2):
#         emb1 = np.array(self.model.encode(ent1))
#         emb2 = np.array(self.model.encode(ent2))
#         return cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
#
#     def extract_entities(self, entities):
#         # allowed_entity_types = {'LAW', 'ORG', 'PRODUCT', 'EVENT'}
#         allowed_entity_types = {'LAW'}
#         entity_set = set()
#         if isinstance(entities, str) and entities.strip():
#             for entity in entities.split('; '):
#                 entity_text, entity_type = entity.rsplit(' ', 1)
#                 entity_text = entity_text.strip()
#                 entity_type = entity_type.strip('()')
#                 if entity_type in allowed_entity_types:
#                     entity_set.add(entity_text)
#         return entity_set
#
#     def store_representative_sentence(self):
#         cluster_centroids = {}
#
#         for cluster in self.issues_df['cluster'].unique():
#             cluster_data = self.issues_df[self.issues_df['cluster'] == cluster]
#             cluster_embeddings = np.vstack(cluster_data['embeddings'].values)
#
#             centroid = np.mean(cluster_embeddings, axis=0)
#             cluster_centroids[cluster] = centroid
#
#             closest_idx = np.argmax(cosine_similarity([centroid], cluster_embeddings)[0])
#             representative_sentence = cluster_data.iloc[closest_idx]['text']
#
#             self.issues_df.loc[
#                 self.issues_df['cluster'] == cluster, 'representative_sentence'] = representative_sentence

#
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer

class IssueClustering:
    def __init__(self, issues_df, casename):
        self.output = f'./arg_mining/data/issue_clustering/{casename}_issue_clusters.csv'
        self.casename = casename
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.issues_df = issues_df
        self.entity_similarity_threshold = 0.85
        self.final_similarity_threshold = 0.8
        self.semantic_weight = 0.7
        self.entity_weight = 0.3
        self.grouped_issues = []

    def process_issues(self):
        valid_judges = self.issues_df['judge'].apply(lambda x: x not in [None, 'None'])
        self.issues_df = self.issues_df[valid_judges]
        self.issues_df['entities'] = self.issues_df['entities'].apply(self.extract_entities)

        similarity_matrix = self.compute_similarity_matrix()
        clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average',
                                             distance_threshold=1 - self.final_similarity_threshold)
        cluster_labels = clustering.fit_predict(1 - similarity_matrix)

        self.issues_df['temp_cluster'] = cluster_labels
        self.filter_clusters()
        self.store_representative_sentence()
        self.issues_df.to_csv(self.output, index=False)

    def compute_similarity_matrix(self):
        num_issues = len(self.issues_df)
        similarity_matrix = np.zeros((num_issues, num_issues))

        for i in range(num_issues):
            for j in range(i + 1, num_issues):
                sim_score = self.compute_combined_similarity(self.issues_df.iloc[i], self.issues_df.iloc[j])
                similarity_matrix[i, j] = sim_score
                similarity_matrix[j, i] = sim_score

        return similarity_matrix

    def compute_combined_similarity(self, issue1, issue2):
        semantic_similarity = cosine_similarity([issue1['embeddings']], [issue2['embeddings']])[0][0]

        if issue1['entities']:
            if issue2['entities']:
                entity_similarity = self.compute_entity_scores(issue1['entities'], issue2['entities'])
            else:
                entity_similarity = 0
            final_score = (self.semantic_weight * semantic_similarity +
                           self.entity_weight * entity_similarity)
        else:
            final_score = semantic_similarity
        return final_score

    def compute_entity_scores(self, entities1, entities2):
        highest_similarity = float('-inf')

        for ent1 in entities1:
            for ent2 in entities2:
                similarity = self.compute_entity_similarity(ent1, ent2)
                highest_similarity = max(highest_similarity, similarity)

        return highest_similarity

    def compute_entity_similarity(self, ent1, ent2):
        emb1 = np.array(self.model.encode(ent1))
        emb2 = np.array(self.model.encode(ent2))
        return cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]

    def extract_entities(self, entities):
        allowed_entity_types = {'LAW', 'ORG', 'PRODUCT', 'EVENT'}
        entity_set = set()
        if isinstance(entities, str) and entities.strip():
            for entity in entities.split('; '):
                entity_text, entity_type = entity.rsplit(' ', 1)
                entity_text = entity_text.strip()
                entity_type = entity_type.strip('()')
                if entity_type in allowed_entity_types:
                    entity_set.add(entity_text)
        return entity_set

    def filter_clusters(self):
        clusters = self.issues_df['temp_cluster'].unique()
        valid_clusters = []

        for cluster in clusters:
            cluster_data = self.issues_df[self.issues_df['temp_cluster'] == cluster]
            if self.is_valid_cluster(cluster_data):
                valid_clusters.append(cluster)

        self.issues_df['cluster'] = self.issues_df['temp_cluster'].apply(
            lambda x: x if x in valid_clusters else -1
        )

        self.issues_df = self.issues_df.drop(columns=['temp_cluster'])

    def is_valid_cluster(self, cluster_data):
        judges = cluster_data['judge'].unique()
        return len(judges) == len(cluster_data)

    def store_representative_sentence(self):
        cluster_centroids = {}

        for cluster in self.issues_df['cluster'].unique():
            cluster_data = self.issues_df[self.issues_df['cluster'] == cluster]
            cluster_embeddings = np.vstack(cluster_data['embeddings'].values)

            centroid = np.mean(cluster_embeddings, axis=0)
            cluster_centroids[cluster] = centroid

            closest_idx = np.argmax(cosine_similarity([centroid], cluster_embeddings)[0])
            representative_sentence = cluster_data.iloc[closest_idx]['text']

            self.issues_df.loc[
                self.issues_df['cluster'] == cluster, 'representative_sentence'] = representative_sentence


