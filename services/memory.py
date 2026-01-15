from db.database import insert_chat

def store_interaction(user_input, assistant_reply):
    insert_chat(user_input, assistant_reply)



#store_interaction("hello", "hi")