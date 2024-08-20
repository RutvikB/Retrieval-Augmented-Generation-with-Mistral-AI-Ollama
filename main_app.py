from web_rag import WEB_LLM
import sys
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    user_input = sys.argv

    print("Running >>", user_input)

    if len(sys.argv) < 2:
        chat_model = WEB_LLM()

        chat_model.load()

        while True:
            query = input("Enter your Query >>> ")

            if len(query) == 0:
                continue
            if query == "/exit":
                break

            chat_model.invoke(query=query)

    elif user_input[1] == "--ingest":
        chat_model = WEB_LLM()

        # Retrieve context files
        with open("./context_file.txt", "r") as f:
            context = f.read()
        
        context_list = context.split('\n')

        chat_model.ingest(context_list)

    else:
        raise SyntaxError("Please check the syntax of the input command")
        


if __name__ == "__main__":
    main()