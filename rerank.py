from sentence_transformers import SentenceTransformer
import scipy
import os


def bert_rerank(context_list: list, query: str, closest_n: int) -> list:
    """ """

    # parameters
    picked_context = []
    log_file = "bot.log"

    # open log file
    if os.path.isfile(log_file):
        log_data = open(log_file, "a")
    else:
        log_data = open(log_file, "w")

    # Embedding with BERT
    embedder = SentenceTransformer("bert-base-nli-mean-tokens")
    corpus_embeddings = embedder.encode(context_list)
    query_embedding = embedder.encode(query)

    # compute distances
    distances = scipy.spatial.distance.cdist(
        [query_embedding], corpus_embeddings, "cosine"
    )[0]

    # pick n closest element
    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])
    for idx, distance in results[0:closest_n]:
        log_data.write(
            f"[RERANK][PICKED] => {context_list[idx].strip()} | SCORE : {1 - distance}"
        )
        picked_context.append(context_list[idx].strip())

    # return selection
    return picked_context


if __name__ == "__main__":

    # parameters
    context_list = [
        "Le système de navigation du X-wing intègre un récepteur stellaire avancé pour une précision maximale lors des sauts en hyperespace.",
        "L'ordinateur de bord du X-wing utilise des coordonnées galactiques pour calculer des itinéraires optimaux tout en évitant les obstacles stellaires.",
        "Les capteurs magnétiques du X-wing assurent un verrouillage précis sur les étoiles, facilitant la navigation dans des secteurs inexplorés.",
        "Le TIE Fighter, doté d'un système de navigation impérial, privilégie la vitesse et la manœuvrabilité pour des missions d'interception rapides.",
        "Les calculateurs de saut du TIE Fighter sont conçus pour des trajets courts, offrant une réactivité exceptionnelle en combat spatial.",
        "Les pingouins utilisent les constellations pour se repérer dans l'immensité glaciale de l'Antarctique, ajustant leur trajectoire en fonction des étoiles.",
        "L'observation du ciel nocturne permet aux pingouins de déterminer la position du pôle Sud, assurant une orientation précise lors de leurs déplacements.",
    ]
    query = "Comment calibré le système de navigation de mon X-wing pour un saut en hyperespace ?"

    # call function
    m = bert_rerank(context_list, query, 3)
    print(m)
