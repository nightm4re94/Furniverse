import requests
import de_core_news_sm
import language_tool_python

import textdescriptives as td
import pandas as pd

from sentence_transformers import SentenceTransformer, util
from PIL import Image
from tqdm import tqdm
from loguru import logger
from germansentiment import SentimentModel

CATEGORIES = ["bed", "bench", "cabinet", "chair", "chest", "couch", "decoration", "desk", "diningtable", "electronics", "hobby", "shelf", "stool", "tableware", "wardrobe"]
ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
PERSPECTIVES = ["down", "up", "frontal"]
MODELS_PER_CATEGORY = 15


def download_databases(api_key, url, output_dir):
    """
    Download contents of the Furniverse databases to a local directory.
    :param api_key: Cloudant api key
    :param url: Cloudant service url
    :param output_dir: Directory in which the downloaded files will be stored.
    :return:
    """
    logger.info(f"Initializing Cloudant connection...")
    authenticator = IAMAuthenticator(api_key)
    service = CloudantV1(authenticator=authenticator)
    service.set_service_url(url)
    service.enable_retries(max_retries=5, retry_interval=0.5)

    logger.info(f"Downloading Furniverse data...")
    for db in ["annotators", "descriptions", "feedback"]:
        logger.info(f"Downloading {db}...")
        all_docs = service.post_all_docs(
            db=db,
            include_docs=True,
        ).get_result()

        df = pd.DataFrame()
        for row in tqdm(all_docs['rows']):
            if row['id'].startswith('_design'):
                continue
            doc_dict = row['doc']
            for k, v in doc_dict.items():
                if not isinstance(v, int) and not v:
                    doc_dict[k] = None
                elif isinstance(v, list):
                    doc_dict[k] = ','.join(v)
                if isinstance(doc_dict[k], str):
                    doc_dict[k] = doc_dict[k].replace('\n', '').replace('\t', ' ')
                    if not any(c.isalnum() for c in doc_dict[k]) and not k.startswith(("_", "$")):
                        doc_dict[k] = None
            doc_df = pd.DataFrame(doc_dict, index=[0])
            df = df.append(doc_df)
        df.to_csv(f"{output_dir}/{db}.csv", sep="\t", header=True, encoding="utf-8", index=False)
    logger.info(f"Downloading models...")
    all_docs = service.post_all_docs(
        db="models",
        include_docs=True,
    ).get_result()

    # Special treatment for the models database due to nested structure.
    models_df = pd.DataFrame()
    for row in tqdm(all_docs['rows']):
        if row['id'].startswith('_design'):
            continue
        doc_dict = row['doc']
        for k, v in doc_dict.items():
            if not isinstance(v, int) and not v:
                doc_dict[k] = None
            elif isinstance(v, list):
                doc_dict[k] = ','.join(v)
            if isinstance(doc_dict[k], str):
                doc_dict[k] = doc_dict[k].replace('\n', '').replace('\t', ' ')
                if not any(c.isalnum() for c in doc_dict[k]) and not k.startswith(("_", "$")):
                    doc_dict[k] = None
        if not doc_dict['autoTags']:
            continue
        filtered_auto_tags = []
        for tag in doc_dict['autoTags']:
            filtered_auto_tags.append(f"{tag}:{doc_dict['autoTags'][tag]}")
        doc_dict['autoTags'] = ','.join(filtered_auto_tags)
        doc_df = pd.json_normalize(doc_dict, sep='>>')
        models_df = models_df.append(doc_df)
    models_df.to_csv(f"{output_dir}/models.csv", sep="\t", header=True, encoding="utf-8", index=False)


def spellcheck_sentence(sentence, spellchecker, mode):
    """
    Checks a sentence for typography errors using a LanguageTool instance.
    :param sentence: The original sentence.
    :param spellchecker: The LanguageTool instance.
    :param mode: The description mode.
    :return: The number of typography errors detected, and a version of the original sentence with corrected spelling.
    """
    sentence = sentence.strip(" \t\n,")
    if not any(sentence.endswith(p) for p in string.punctuation):
        sentence += "."
    errors = spellchecker.check(sentence)
    error_count = len(errors)
    suggestion = sentence if error_count == 0 else spellchecker.correct(sentence)
    return pd.Series([error_count, suggestion], index=[f"{mode}_spelling_errors", f"{mode}_spelling_suggested"])


def sentiment_to_number(sentiment):
    """
    Translate a sentiment string to a numeric value.
    :param sentiment: One of 'positive', 'neutral', 'negative'
    :return: 1 if 'positive', 0 if 'neutral', -1 if "negative"
    """
    if sentiment == "positive":
        return 1
    elif sentiment == "neutral":
        return 0
    elif sentiment == "negative":
        return -1


def parse_descriptions(output_dir, affective_norms_file):
    """
    Process the downloaded descriptions by applying spellchecking, assessing readability and surface statistics, analyzing sentiment, and evaluating affective norms.
    :param output_dir: Directory containing the downloaded descriptions. The processed descriptions and stats will also be stored here.
    :param affective_norms_file: Path to a file with affective norms downloaded from https://www.ims.uni-stuttgart.de/en/research/resources/experiment-data/affective-norms/
    :return:
    """
    logger.info(f"Initializing nlp pipeline...")
    nlp = de_core_news_sm.load()
    nlp.add_pipe("textdescriptives")

    spellchecker = language_tool_python.LanguageTool('de-DE')
    sentiment_analyzer = SentimentModel()

    descriptions_df = pd.read_csv(f"{output_dir}/descriptions.csv", sep="\t", encoding="utf-8")
    affective_norms_df = pd.read_csv(affective_norms_file, sep="\t", encoding="utf-8", index_col='Word')
    for mode in ["literal", "sentimental"]:
        logger.info(f"Processing {mode} descriptions...")
        logger.info(f"Spellchecking...")
        spellchecked = descriptions_df[mode].apply(spellcheck_sentence, args=(spellchecker, mode))
        descriptions_df = descriptions_df.join(spellchecked)

        logger.info(f"Analysing sentiment...")
        descriptions_df[f"{mode}_sentiment"] = sentiment_analyzer.predict_sentiment(descriptions_df[f"{mode}"])
        descriptions_df[f"{mode}_sentiment"] = descriptions_df[f"{mode}_sentiment"].apply(sentiment_to_number)

        logger.info(f"Assessing readability and affective variables...")
        docs = nlp.pipe(descriptions_df[f"{mode}_spelling_suggested"])
        stats_df = pd.DataFrame()
        for doc in docs:
            readability_df = td.extract_df(doc)
            affective_values = affective_analysis(doc, affective_norms_df)
            readability_df.insert(1, "concreteness", affective_values[0])
            readability_df.insert(2, "valence", affective_values[1])
            readability_df.insert(3, "arousal", affective_values[2])
            readability_df.insert(4, "imageability", affective_values[3])
            stats_df = stats_df.append(readability_df)
        stats_df = stats_df.set_index(descriptions_df["_id"]).drop('smog', 1)
        stats_df.to_csv(f"{output_dir}/{mode}_stats.csv", sep="\t", encoding="utf-8")
    descriptions_df.to_csv(f"{output_dir}/descriptions.csv", sep="\t", header=True, encoding="utf-8", index=False)
    spellchecker.close()


def affective_analysis(doc, lexicon):
    """
    Evaluate affective norms concreteness, valence, arousal, and imageability for a given CoNLL-U formatted document based on the norms
    by Köper / Schulte im Walde (2016): https://www.ims.uni-stuttgart.de/en/research/resources/experiment-data/affective-norms/
    :param doc: CoNLL-U formatted text.
    :param lexicon: DataFrame with the affective norms.
    :return: list of affective norms for the document.
    """
    concreteness = 0
    valence = 0
    arousal = 0
    imageability = 0
    words_in_lexicon = 0
    for tok in doc:
        if tok.lemma_ in lexicon.index:
            words_in_lexicon += 1
            concreteness += lexicon["AbstConc"][tok.lemma_]
            valence += lexicon["Val"][tok.lemma_]
            arousal += lexicon["Arou"][tok.lemma_]
            imageability += lexicon["IMG"][tok.lemma_]
    if words_in_lexicon == 0:
        concreteness = None
        valence = None
        arousal = None
        imageability = None
    else:
        concreteness /= words_in_lexicon
        valence /= words_in_lexicon
        arousal /= words_in_lexicon
        imageability /= words_in_lexicon
    return [concreteness, valence, arousal, imageability]


def load_image(url_or_path):
    """
    Open an image from either a web adress or local path.
    This enables using the Furniverse resources either from the Furniverse web repository, or from a downloaded folder.
    :param url_or_path: string containing the image's web address or path on the file system.
    :return: the loaded image
    """
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return Image.open(requests.get(url_or_path, stream=True).raw)
    else:
        return Image.open(url_or_path)


def extract_features(images_dir, stats_dir):
    """
    Extract feature vectors for all images, categories, and descriptions in Furniverse
    :param images_dir: string containing the parent directory or parent web address where all images are located.
    :param stats_dir: string containing the parent directory where the descriptions and models files are located, and where the embeddings will be stored.
    :return:
    """
    logger.info(f"Initializing Image Model...")
    img_model = SentenceTransformer('clip-ViT-B-32')
    logger.info(f"Initializing Text Model...")
    text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
    logger.info(f"Reading metadata and descriptions...")
    descriptions_df = pd.read_csv(f"{stats_dir}/descriptions.csv", sep="\t", encoding="utf-8")

    logger.info(f"Extracting features...")
    embeddings_df = pd.DataFrame()

    category_embeddings = pd.DataFrame(text_model.encode(CATEGORIES))
    category_embeddings.insert(0, "_id", [f"{cat}_category" for cat in CATEGORIES])
    embeddings_df = embeddings_df.append(category_embeddings)

    for (ind, cat) in tqdm([(i, cat) for i in range(1, MODELS_PER_CATEGORY + 1) for cat in CATEGORIES]):
        furnishing = f"{cat}-{ind}"
        img_paths = [f"{images_dir}/{furnishing}_{angle}_{perspective}.png" for angle in ANGLES for perspective in PERSPECTIVES]
        img_embeddings = pd.DataFrame(img_model.encode([load_image(p).convert('RGB') for p in img_paths]))

        literal_descriptions = descriptions_df[descriptions_df["model"] == furnishing]["literal"]
        sentimental_descriptions = descriptions_df[descriptions_df["model"] == furnishing]["sentimental"]
        literal_embeddings = pd.DataFrame(text_model.encode(literal_descriptions.tolist()))
        sentimental_embeddings = pd.DataFrame(text_model.encode(sentimental_descriptions.tolist()))

        img_embeddings.insert(0, "_id", [f"{p.split('/')[-1]}" for p in img_paths])
        literal_embeddings.insert(0, "_id", [f"{furnishing}_literal_{d}" for d in descriptions_df[descriptions_df["model"] == furnishing]["_id"]])
        sentimental_embeddings.insert(0, "_id", [f"{furnishing}_sentimental_{d}" for d in descriptions_df[descriptions_df["model"] == furnishing]["_id"]])

        embeddings_df = embeddings_df.append(img_embeddings)
        embeddings_df = embeddings_df.append(literal_embeddings)
        embeddings_df = embeddings_df.append(sentimental_embeddings)

    embeddings_df.to_csv(f"{stats_dir}/embeddings.csv", sep="\t", header=True, encoding="utf-8", index=False)
