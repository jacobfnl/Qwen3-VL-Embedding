# MSRVTT Text-to-Video Retrieval Dataset Export

## Loading the dataset

```python
from datasets import load_from_disk

dataset = load_from_disk("data/examples/msrvtt")

# Access subsets
queries = dataset["queries"]
corpus = dataset["corpus"]
relevant_docs = dataset["relevant_docs"]
```

## Dataset Structure

- **queries**: 10 samples (text captions)
  - query_id: unique query identifier
  - text: video caption/description
  - instruction: retrieval task instruction ("Find a video that contains the following visual content.")

- **corpus**: 10 samples (videos)
  - corpus_id: unique corpus identifier (c_{video_id})
  - video_id: original video identifier
  - video_paths: list of video frame paths (relative to dataset root)

- **relevant_docs**: 10 samples
  - query_id: corresponding query ID
  - corpus_ids: list containing the relevant video corpus ID

## Task Description

**Task Type**: Text-to-Video Retrieval

**Instruction**: Find a video that contains the following visual content.

This is a text-to-video retrieval task where text queries (captions) are matched to relevant videos from the corpus.

## Videos

Video frames are stored in: `videos/`

Each video is represented by 64 frames extracted from up to 64 saved frames.

## Statistics

- Total unique videos: 10
- Total text queries: 10
- Queries per video: 1.00 (average)

## Example Query

```python
query = queries[0]
print(f"Query ID: {query['query_id']}")
print(f"Text: {query['text']}")
print(f"Instruction: {query['instruction']}")

# Find relevant video
relevant = relevant_docs[0]
corpus_id = relevant['corpus_ids'][0]
video = corpus.filter(lambda x: x['corpus_id'] == corpus_id)[0]
print(f"Relevant video: {video['video_id']}")
print(f"Video frames: {len(video['video_paths'])} frames")
```
