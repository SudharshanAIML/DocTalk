from fastapi import FastAPI, HTTPException
from dp.mongo.py import init_dp

app = FastAPI()




# this is the api flow for reindexing 
# delete_document(user_id, file_id)
# delete_chunks_by_file(user_id, file_id)
# rebuild_user_index(user_id)

