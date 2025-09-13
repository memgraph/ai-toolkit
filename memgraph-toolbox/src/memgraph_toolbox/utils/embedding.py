import torch  # TODO(gitbuda): Huge import for a very small module.

from sentence_transformers import SentenceTransformer
from typing import Iterable, List

# TODO(gitbuda): Offer the out-of-the-box parallelization, both on CPU and GPU.
## option#1
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
# model = torch.nn.DataParallel(model)
# model = model.cuda()
# sentences = ["Hello world", "Graph databases are fast"]
# embeddings = model.module.encode(sentences, batch_size=64, convert_to_tensor=True)
# print(embeddings.shape)
## option#2
# import torch
# from multiprocessing import Pool
# def encode_on_device(device_id, sentences):
#     model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=f"cuda:{device_id}")
#     return model.encode(sentences, batch_size=64, convert_to_tensor=True)
# sentences = ["text 1", "text 2", ..., "text N"]
# chunks = [sentences[i::2] for i in range(2)]  # split across 2 GPUs
# with Pool(2) as p:
#     results = p.starmap(encode_on_device, [(0, chunks[0]), (1, chunks[1])])
# embeddings = torch.cat(results)
# print(embeddings.shape)


# NOTE: HF_TOKEN has to be set in the environment variables.
def get_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    return model


def encode(texts: Iterable[List[str]], model: SentenceTransformer):
    for batch in texts:
        yield model.encode(batch)
