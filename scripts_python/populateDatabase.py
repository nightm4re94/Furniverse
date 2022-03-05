import time
from pathlib import Path

import pycouchdb
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibmcloudant import CloudantV1
from ibmcloudant.cloudant_v1 import Document
from loguru import logger

import pandas as pd
from pandas.errors import EmptyDataError
from tqdm import tqdm


def populate_cloudant_database(api_key, url, data_file_path, auto_tags_dir):
    """
    If you are hosting Furniverse via Cloudant, use this method to populate your models database with metadata collected in a tab-separated text file.
    :param api_key: The cloudant API key used for authentication (needs write access).
    :param url: The permanent URL of the Furniverse Cloudant resource.
    :param data_file_path: The path to the tab-separated metadata file.
    :param auto_tags_dir: The path to the directory containing automatically generated tags for each model.
    """
    p = Path(data_file_path)
    # Initialize Cloudant connection
    authenticator = IAMAuthenticator(api_key)
    service = CloudantV1(authenticator=authenticator)
    service.set_service_url(url)
    service.enable_retries(max_retries=5, retry_interval=0.5)
    db_name = "models"
    # Process tab-separated metadata file
    df = pd.read_csv(p, delimiter="\t")
    for index, row in tqdm(df.iterrows()):
        # Skip rows if the corresponding entry is already in the database
        if len(service.post_all_docs(db_name, key=row['ID']).get_result()['rows']) > 0:
            logger.info(f"Skipped database entry creation for model '{row['ID']}' (already exists).")
            time.sleep(.5)
            continue
        # Try to read automatically predicted tags (if that fails, keep the autoTags block empty)
        try:
            auto_tags_file = Path(f"{auto_tags_dir}/{row['ID']}.pred")
            auto_tags = pd.read_csv(auto_tags_file, delimiter=":", header=None)
            auto_tags_dict = {e[0]: e[1] for i, e in auto_tags.iterrows()}
        except EmptyDataError:
            auto_tags_dict = {}
        doc = {"_id": row['ID'],
               "$doctype": "model",
               "title": row['Title'],
               "category": row['Category'],
               "autoTags": auto_tags_dict,
               "tags": row['Tags'].split(",") if isinstance(row['Tags'], str) else [],
               "description": row['Description'] if isinstance(row['Description'], str) else "",
               "url": row['URL'],
               "copyright": {
                   "author": row['Author'],
                   "license": row['LicenseName'],
                   "url": row['LicenseURL']
               }
               }
        # Post the data dictionary in the database
        document = Document.from_dict(doc)
        service.post_document(db=db_name, document=document).get_result()
        logger.info(f"Created database entry '{row['ID']}'.")


def populate_couchdb_database(username, password, url, data_file_path, auto_tags_dir):
    """
    If you are hosting Furniverse via CouchDB, use this method to populate your models database with metadata collected in a tab-separated text file.
    In this simple example, we are authenticating via username and password, but you should consider a safer authentication method in order not to expose user data.
    :param username: The CouchDB user name (needs write access).
    :param password: The CouchDB user password.
    :param url: The permanent URL of the CouchDB server.
    :param data_file_path: The path to the tab-separated metadata file.
    :param auto_tags_dir: The path to the directory containing automatically generated tags for each model.
    """
    p = Path(data_file_path)
    server = pycouchdb.Server(f"""http://{username}:{password}@{url.replace("http://", "")}""")
    model_db = server.database("models")
    df = pd.read_csv(p, delimiter="\t")
    for index, row in tqdm(df.iterrows()):
        if f"{row['ID']}" in model_db:
            logger.info(f"Skipped database entry creation for model '{row['ID']}' (entry already exists).")
            continue
        try:
            auto_tags_file = Path(f"{auto_tags_dir}/{row['ID']}.pred")
            auto_tags = pd.read_csv(auto_tags_file, delimiter=":", header=None)
            auto_tags_dict = {e[0]: e[1] for i, e in auto_tags.iterrows()}
        except EmptyDataError:
            auto_tags_dict = dict()
        doc = {"_id": row['ID'],
               "$doctype": "model",
               "title": row['Title'],
               "category": row['Category'],
               "autoTags": auto_tags_dict,
               "tags": row['Tags'].split(",") if isinstance(row['Tags'], str) else [],
               "description": row['Description'] if isinstance(row['Description'], str) else "",
               "url": row['URL'],
               "copyright": {
                   "author": row['Author'],
                   "license": row['LicenseName'],
                   "url": row['LicenseURL']
               }
               }
        model_db.save(doc)
