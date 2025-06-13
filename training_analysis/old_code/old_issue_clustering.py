# Attempt on granular semantic similarity:

# def calculate_weighted_similarity(self, sentence1, sentence2):
    #     # split into components and generate embeddings
    #     components1 = self.split_sentence(sentence1)
    #     components2 = self.split_sentence(sentence2)
    #     embeddings1 = self.model.encode(components1)
    #     embeddings2 = self.model.encode(components2)
    #
    #     similarity_matrix = self.compute_component_similarity(embeddings1, embeddings2)
    #
    #     # calculate how much of the sentence the component makes up to get its weight
    #     weight1 = np.array([len(comp) for comp in components1]) / sum(len(comp) for comp in components1)
    #     weight2 = np.array([len(comp) for comp in components2]) / sum(len(comp) for comp in components2)
    #
    #     weighted_similarity = np.sum(similarity_matrix * weight1[:, None] * weight2[None, :])
    #
    #     max_similarity = np.max(similarity_matrix)
    #     if max_similarity > 0:
    #         weighted_similarity /= max_similarity
    #
    #     return weighted_similarity
    #
    # def split_sentence(self, sentence):
    #     components = re.split(r'(\s*[,;."\'()]\s*)', sentence)
    #     return [comp.strip() for comp in components if comp.strip()]
    #
    # def compute_component_similarity(self, embeddings1, embeddings2):
    #     # Compute pairwise cosine similarity between the embeddings of the two sets of components
    #     similarity_matrix = cosine_similarity(embeddings1, embeddings2)
    #     return similarity_matrix







# ISSUE clustering based on NER-Labelling
class EntitySimilarity:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def analyse_issues_entities(self, casename, issues_df):
        print('\nIssues Clustered on Contained Entity Similarity:\n')

        # issues_df = self.load_ner_features(casename)
        # issues_df = issues_df.reset_index(drop=True)

        sentence_entities = [self.extract_entities(entities) for entities in issues_df['entities']]
        similarity_matrix = self.create_similarity_matrix(sentence_entities)
        clusters = self.create_clusters(similarity_matrix)

        issues_df['cluster'] = [cluster for cluster in clusters]

        self.save_clusters(issues_df, casename)

        num_clusters = len(set(clusters))
        print(f"Number of distinct issues: {num_clusters}")
        self.print_clusters(issues_df)

    # def load_ner_features(self, casename):
    #     entity_file_path = './summarydata-spacy/UKHL_' + casename + '.csv'
    #     entity_df = pd.read_csv(entity_file_path)
    #     arg_file_path = './arg_mining/data/arg_role_labelled/' + casename + '.csv'
    #     arg_df = pd.read_csv(arg_file_path)
    #
    #     arg_df['entities'] = entity_df['entities']
    #     issues_df = arg_df[arg_df['arg_role'] == 'ISSUE']
    #     return issues_df

    def extract_entities(self, entities):
        allowed_entity_types = {'LAW', 'ORG', 'PRODUCT', 'EVENT'}
        entity_set = set()
        if isinstance(entities, str) and entities.strip():
            for entity in entities.split('; '):
                entity_text, entity_type = entity.rsplit(' ', 1)
                entity_text = entity_text.strip()
                entity_type = entity_type.strip('()')
                if entity_type in allowed_entity_types:  # filter by allowed types
                    entity_set.add((entity_text, entity_type))
        return entity_set

    def create_similarity_matrix(self, sentence_entities):
        n = len(sentence_entities)
        similarity_matrix = [[0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                if self.has_common_entities(sentence_entities[i], sentence_entities[j]):
                    similarity_matrix[i][j] = similarity_matrix[j][i] = 1

        return similarity_matrix

    def has_common_entities(self, entities1, entities2):
        # any exact matches?
        if not entities1.intersection(entities2) == set():
            return True

        # any sufficiently similar matches?
        for ent1 in entities1:
            for ent2 in entities2:
                if self.compute_entity_similarity(ent1, ent2) > 0.9:
                    return True
        return False

    def compute_entity_similarity(self, ent1, ent2):
        emb1 = self.model.encode(ent1)
        emb2 = self.model.encode(ent2)
        emb1 = np.array(emb1).reshape(1, -1)
        emb2 = np.array(emb2).reshape(1, -1)
        result = cosine_similarity(emb1, emb2)[0][0]
        # result = self.model.similarity(emb1, emb2)
        # print(f'Comparing {ent1} and {ent2} = a similarity of {result}\n')
        return result

    def create_clusters(self, similarity_matrix):
        n = len(similarity_matrix)
        visited = [False] * n
        clusters = [-1] * n

        def dfs(node, cluster_id):
            stack = [node]
            while stack:
                v = stack.pop()
                if not visited[v]:
                    visited[v] = True
                    clusters[v] = cluster_id
                    for i, connected in enumerate(similarity_matrix[v]):
                        if connected and not visited[i]:
                            stack.append(i)

        cluster_id = 0
        for i in range(n):
            if not visited[i]:
                dfs(i, cluster_id)
                cluster_id += 1

        return clusters

    def save_clusters(self, df, casename):
        output_file_path = './arg_mining/data/issue_clustering/' + casename + '_issue_clusters_ENTITIES.csv'
        cluster_df = df[['sentence_id', 'judge', 'mj', 'text', 'entities', 'cluster']]
        cluster_df.to_csv(output_file_path, index=False)
        print(f"Cluster results saved to {output_file_path}")

    def print_clusters(self, df):
        allowed_entity_types = {'LAW', 'ORG', 'PRODUCT', 'EVENT'}
        clusters = df.groupby('cluster')

        for cluster_id, group in clusters:
            print(f"Cluster {cluster_id}:")

            entity_sets = []

            for _, row in group.iterrows():
                print(f" - Text: {row['text']}")
                print(f"   Entities: {row['entities']}")

                if isinstance(row['entities'], str) and row['entities'].strip():
                    entities_set = set()
                    for entity in row['entities'].split('; '):
                        entity_text, entity_type = entity.rsplit(' ', 1)
                        entity_text = entity_text.strip()
                        entity_type = entity_type.strip('()')
                        if entity_type in allowed_entity_types:  # FILTER BY ALLOWED TYPES
                            entities_set.add((entity_text, entity_type))
                    entity_sets.append(entities_set)

            if entity_sets:
                common_entities = set.intersection(*entity_sets)
                if common_entities:
                    print(f"Common Entities: {', '.join(f'{text} ({type})' for text, type in common_entities)}")
                else:
                    print("No common entities.")
            else:
                print("No entities available.")

            print("\n")


# ISSUE CLUSTERING BASED ON SIMILARITY OF EMBEDDINGS
class EmbeddingSimilarity:
    def analyse_issues_embeddings(self, casename, issues_df):
        print('\nIssues Clustered on Embedding Similarity:\n')
        # issues_df = self.load_embeddings(casename)
        # similarity_df = self.compute_similarity(issues_df)
        labels = self.cluster_embeddings(issues_df['embeddings'].tolist(), threshold=0.75)
        num_clusters = self.determine_number_of_clusters(labels)
        print(f"Number of distinct issues: {num_clusters}")
        representative_sentences = self.find_representative_sentences(issues_df, labels)
        self.save_clusters(issues_df, labels, casename, representative_sentences)
        self.print_clusters(issues_df, labels, representative_sentences)
    #
    # def load_embeddings(self, casename):
    #     file_path = './arg_mining/data/arg_role_labelled/' + casename + '.csv'
    #     df = pd.read_csv(file_path)
    #
    #     df['embeddings'] = df['embeddings'].apply(lambda x: ast.literal_eval(x))
    #
    #     issues_df = df[df['arg_role'] == 'ISSUE']
    #     issues_df = issues_df.reset_index(drop=True)
    #     print(issues_df)
    #     return issues_df

    # def compute_similarity(self, df):
    #     embeddings = df['embeddings'].tolist()
    #     similarity_matrix = cosine_similarity(embeddings)
    #     similarity_df = pd.DataFrame(similarity_matrix, index=df['text'], columns=df['text'])
    #     return similarity_df

    def cluster_embeddings(self, embeddings, threshold=0.75):
        # for emb in embeddings:
        #     print(np.array(emb).shape)
        embeddings_array = np.array(embeddings)
        clustering = AgglomerativeClustering(n_clusters=None, affinity='cosine', linkage='average',
                                             distance_threshold=1 - threshold)
        clustering.fit(embeddings_array)
        return clustering.labels_

    def save_clusters(self, df, labels, casename, representative_sentences):
        df['cluster'] = labels
        df['embedding_representative'] = df['cluster'].map(representative_sentences)
        output_file_path = './arg_mining/data/issue_clustering/' + casename + '_issue_clusters_EMBEDDINGS.csv'
        cluster_df = df[['sentence_id', 'judge', 'mj', 'text', 'embedding_representative', 'cluster']]
        cluster_df.to_csv(output_file_path, index=False)
        print(f"Cluster results saved to {output_file_path}")

    def determine_number_of_clusters(self, labels):
        return len(set(labels))

    def find_representative_sentences(self, df, labels):
        df['cluster'] = labels
        clusters = df.groupby('cluster')
        representative_sentences = {}

        for cluster_id, group in clusters:
            sentences = group['text'].tolist()
            embeddings = np.array(group['embeddings'].tolist())

            centroid = np.mean(embeddings, axis=0)
            similarities = cosine_similarity([centroid], embeddings)[0]

            representative_index = np.argmax(similarities)
            representative_sentences[cluster_id] = sentences[representative_index]

        return representative_sentences

    def print_clusters(self, df, labels, representative_sentences):
        df['cluster'] = labels
        clusters = df.groupby('cluster')

        for cluster_id, group in clusters:
            print(f"Cluster {cluster_id}:")
            for text in group['text']:
                print(f" - {text}")
            print("\n")
            print(f"Cluster Representative Sentence: {representative_sentences[cluster_id]}")
            print("\n")


class ClusterCombiner:
    def __init__(self, casename):
        entity_file = './arg_mining/data/issue_clustering/' + casename + '_issue_clusters_ENTITIES.csv'
        embeddings_file = './arg_mining/data/issue_clustering/' + casename + '_issue_clusters_EMBEDDINGS.csv'
        self.entity_df = pd.read_csv(entity_file)
        self.embedding_df = pd.read_csv(embeddings_file)

    def create_probability_matrix(self):
        texts = self.entity_df['text'].tolist()
        num_texts = len(texts)

        # Initialize probability matrix
        probability_matrix = np.zeros((num_texts, num_texts))

        # Map text to indices
        text_to_index = {text: i for i, text in enumerate(texts)}

        # Map clusters
        entity_clusters = self.entity_df.set_index('text')['cluster'].to_dict()
        embedding_clusters = self.embedding_df.set_index('text')['cluster'].to_dict()

        # Calculate similarity based on entity clustering and embedding clustering
        for text1 in texts:
            for text2 in texts:
                if text1 != text2:
                    entity_cluster1 = entity_clusters.get(text1, -1)
                    entity_cluster2 = entity_clusters.get(text2, -1)
                    embedding_cluster1 = embedding_clusters.get(text1, -1)
                    embedding_cluster2 = embedding_clusters.get(text2, -1)

                    # Increment probability if both clusters match
                    if entity_cluster1 == entity_cluster2 and entity_cluster1 != -1:
                        probability_matrix[text_to_index[text1], text_to_index[text2]] += 1
                    if embedding_cluster1 == embedding_cluster2 and embedding_cluster1 != -1:
                        probability_matrix[text_to_index[text1], text_to_index[text2]] += 1

        # normalise by total number of clusters
        max_probabilities = np.max(probability_matrix, axis=1, keepdims=True)
        max_probabilities[max_probabilities == 0] = 1  # Avoid division by zero
        probability_matrix /= max_probabilities

        return probability_matrix

    def combine_clusters(self, probability_matrix, threshold=0.75):
        num_texts = probability_matrix.shape[0]
        visited = [False] * num_texts
        new_clusters = [-1] * num_texts
        cluster_id = 0

        def dfs(text_index, new_cluster_id):
            stack = [text_index]
            while stack:
                current_index = stack.pop()
                if not visited[current_index]:
                    visited[current_index] = True
                    new_clusters[current_index] = new_cluster_id
                    for neighbor_index in range(num_texts):
                        if probability_matrix[current_index, neighbor_index] > threshold \
                                and not visited[neighbor_index]:
                            stack.append(neighbor_index)

        for i in range(num_texts):
            if not visited[i]:
                dfs(i, cluster_id)
                cluster_id += 1

        return new_clusters

    def calculate_entity_count(self, entities):
        """Count the number of entities in a statement that are of the specified types."""
        allowed_entity_types = {'LAW', 'ORG', 'PRODUCT', 'EVENT'}

        if isinstance(entities, str):
            entity_list = entities.split('; ')
            filtered_entities = [
                entity for entity in entity_list
                if entity.split(' ')[-1].strip('()') in allowed_entity_types
            ]
            return len(filtered_entities)

        return 0

    def get_representative_scores(self, df):
        """Calculate scores for each issue statement based on multiple factors."""
        df['length_score'] = df['text'].apply(len)
        df['entity_count'] = df['entities'].apply(self.calculate_entity_count)
        df['embedding_rep'] = df['text'].apply(
            lambda x: 1 if x in self.embedding_df[self.embedding_df['embedding_representative'] == True][
                'text'].values else 0)

        df['normalised_length_score'] = df['length_score'] / df['length_score'].max()

        max_value = df['entity_count'].max()
        if max_value == 0:
            # avoid division by zero
            df['normalised_entity_count'] = 0.0
        else:
            df['normalised_entity_count'] = df['entity_count'] / max_value

        weight_length = 0.2
        weight_embedding = 0.4
        weight_entities = 0.4

        # Calculate final score
        df['final_score'] = (weight_length * df['normalised_length_score'] +
                             weight_embedding * df['embedding_rep'] +
                             weight_entities * df['normalised_entity_count'])

        return df

    def find_most_representative_sentences(self, df):
        df['combined_cluster'] = df['combined_cluster'].astype(int)
        scores_df = self.get_representative_scores(df)
        representative_sentences = scores_df.loc[scores_df.groupby('combined_cluster')['final_score'].idxmax()]
        return representative_sentences[['combined_cluster', 'text', 'final_score']]

    def merge_and_save_clusters(self, casename):
        probability_matrix = self.create_probability_matrix()
        combined_clusters = self.combine_clusters(probability_matrix)

        self.entity_df['combined_cluster'] = combined_clusters

        representative_sentences = self.find_most_representative_sentences(self.entity_df)

        self.entity_df['representative_sentence'] = self.entity_df['combined_cluster'].map(
            representative_sentences.set_index('combined_cluster')['text']
        )

        output_file_path = f'./arg_mining/data/issue_clustering/{casename}_combined_clusters.csv'
        self.entity_df.to_csv(output_file_path, index=False)
        print(f"Combined cluster results saved to {output_file_path}")
        self.print_clusters(self.entity_df)

    def print_clusters(self, df):
        clusters = df.groupby('combined_cluster')

        print("Combined Clustering Results:")
        for cluster_id, group in clusters:
            print(f"Cluster {cluster_id}:")
            for text in group['text']:
                print(f" - {text}")
            print("\n")
            representative_sentence = group['representative_sentence'].iloc[0] if not group[
                'representative_sentence'].isna().all() else "None"
            print(f"Cluster Representative Sentence: {representative_sentence}")
            print("\n")