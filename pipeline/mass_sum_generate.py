from IRC_TripleSummaryPipeline import IRC_pipeline

def process_urls(file_path):
    pipeline = IRC_pipeline()
    with open(file_path, 'r') as file:
        for line in file:
            url = line.strip()
            if url:
                try:
                    print(f"Processing URL: {url}")
                    pipeline.begin(url)
                except Exception as e:
                    print(f"Error processing URL {url}: {e}")


process_urls('urls.txt')
