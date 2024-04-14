import os
from typing import List, Tuple, Optional

from openai import OpenAI
import tiktoken
from tqdm import tqdm
from dotenv import load_dotenv

import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Now you can safely access the OPENAI_API_KEY environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_chat_completion(messages, model='gpt-3.5-turbo'):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content


def tokenize(text: str) -> List[str]:
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    return encoding.encode(text)


# This function chunks a text into smaller pieces based on a maximum token count and a delimiter.
def chunk_on_delimiter(input_string: str,
                       max_tokens: int, delimiter: str) -> List[str]:
    chunks = input_string.split(delimiter)
    combined_chunks, _, dropped_chunk_count = combine_chunks_with_no_minimum(
        chunks, max_tokens, chunk_delimiter=delimiter, add_ellipsis_for_overflow=True
    )
    if dropped_chunk_count > 0:
        print(f"warning: {dropped_chunk_count} chunks were dropped due to overflow")
    combined_chunks = [f"{chunk}{delimiter}" for chunk in combined_chunks]
    return combined_chunks


# This function combines text chunks into larger blocks without exceeding a specified token count. It returns the combined text blocks, their original indices, and the count of chunks dropped due to overflow.
def combine_chunks_with_no_minimum(
        chunks: List[str],
        max_tokens: int,
        chunk_delimiter="\n\n",
        header: Optional[str] = None,
        add_ellipsis_for_overflow=False,
) -> Tuple[List[str], List[int]]:
    dropped_chunk_count = 0
    output = []  # list to hold the final combined chunks
    output_indices = []  # list to hold the indices of the final combined chunks
    candidate = (
        [] if header is None else [header]
    )  # list to hold the current combined chunk candidate
    candidate_indices = []
    for chunk_i, chunk in enumerate(chunks):
        chunk_with_header = [chunk] if header is None else [header, chunk]
        if len(tokenize(chunk_delimiter.join(chunk_with_header))) > max_tokens:
            print(f"warning: chunk overflow")
            if (
                    add_ellipsis_for_overflow
                    and len(tokenize(chunk_delimiter.join(candidate + ["..."]))) <= max_tokens
            ):
                candidate.append("...")
                dropped_chunk_count += 1
            continue  # this case would break downstream assumptions
        # estimate token count with the current chunk added
        extended_candidate_token_count = len(tokenize(chunk_delimiter.join(candidate + [chunk])))
        # If the token count exceeds max_tokens, add the current candidate to output and start a new candidate
        if extended_candidate_token_count > max_tokens:
            output.append(chunk_delimiter.join(candidate))
            output_indices.append(candidate_indices)
            candidate = chunk_with_header  # re-initialize candidate
            candidate_indices = [chunk_i]
        # otherwise keep extending the candidate
        else:
            candidate.append(chunk)
            candidate_indices.append(chunk_i)
    # add the remaining candidate to output if it's not empty
    if (header is not None and len(candidate) > 1) or (header is None and len(candidate) > 0):
        output.append(chunk_delimiter.join(candidate))
        output_indices.append(candidate_indices)
    return output, output_indices, dropped_chunk_count


def summarize(text: str,
              detail: float = 0,
              model: str = 'gpt-3.5-turbo',
              additional_instructions: Optional[str] = None,
              minimum_chunk_size: Optional[int] = 500,
              chunk_delimiter: str = ".",
              summarize_recursively=False,
              verbose=False):
    """
    Summarizes a given text by splitting it into chunks, each of which is summarized individually. 
    The level of detail in the summary can be adjusted, and the process can optionally be made recursive.

    Parameters:
    - text (str): The text to be summarized.
    - detail (float, optional): A value between 0 and 1 indicating the desired level of detail in the summary.
      0 leads to a higher level summary, and 1 results in a more detailed summary. Defaults to 0.
    - model (str, optional): The model to use for generating summaries. Defaults to 'gpt-3.5-turbo'.
    - additional_instructions (Optional[str], optional): Additional instructions to provide to the model for customizing summaries.
    - minimum_chunk_size (Optional[int], optional): The minimum size for text chunks. Defaults to 500.
    - chunk_delimiter (str, optional): The delimiter used to split the text into chunks. Defaults to ".".
    - summarize_recursively (bool, optional): If True, summaries are generated recursively, using previous summaries for context.
    - verbose (bool, optional): If True, prints detailed information about the chunking process.

    Returns:
    - str: The final compiled summary of the text.

    The function first determines the number of chunks by interpolating between a minimum and a maximum chunk count based on the `detail` parameter. 
    It then splits the text into chunks and summarizes each chunk. If `summarize_recursively` is True, each summary is based on the previous summaries, 
    adding more context to the summarization process. The function returns a compiled summary of all chunks.
    """

    # check detail is set correctly
    assert 0 <= detail <= 1

    # interpolate the number of chunks based to get specified level of detail
    max_chunks = len(chunk_on_delimiter(text, minimum_chunk_size, chunk_delimiter))
    min_chunks = 1
    num_chunks = int(min_chunks + detail * (max_chunks - min_chunks))

    # adjust chunk_size based on interpolated number of chunks
    document_length = len(tokenize(text))
    chunk_size = max(minimum_chunk_size, document_length // num_chunks)
    text_chunks = chunk_on_delimiter(text, chunk_size, chunk_delimiter)
    if verbose:
        print(f"Splitting the text into {len(text_chunks)} chunks to be summarized.")
        print(f"Chunk lengths are {[len(tokenize(x)) for x in text_chunks]}")

    # set system message
    system_message_content = "Summarize the following text."
    if additional_instructions is not None:
        system_message_content += f"\n\n{additional_instructions}"

    accumulated_summaries = []
    for chunk in tqdm(text_chunks):
        if summarize_recursively and accumulated_summaries:
            # Creating a structured prompt for recursive summarization
            accumulated_summaries_string = '\n\n'.join(accumulated_summaries)
            user_message_content = f"Previous summaries:\n\n{accumulated_summaries_string}\n\nText to summarize next:\n\n{chunk}"
        else:
            # Directly passing the chunk for summarization without recursive context
            user_message_content = chunk

        # Constructing messages based on whether recursive summarization is applied
        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_content}
        ]

        # Assuming this function gets the completion and works as expected
        response = get_chat_completion(messages, model=model)
        accumulated_summaries.append(response)

    # Compile final summary from partial summaries
    final_summary = '\n\n'.join(accumulated_summaries)

    return final_summary

def give_narrative(roadtrip): 
    output = summarize(
        roadtrip, 
        detail=0.4, 
        verbose=True, 
        additional_instructions="Write in elaborate paragraphs about a roadtrip narrative given these attractions. Include high level of detail. Pretend like you're a tour guide."
        )
    print(output)


if __name__ == "__main__": 

    # insert roadtrip cities csv here
    path = "resultFile3.csv"

    cities = pd.read_csv(path, header=None).values.flatten().tolist()
    print(cities)

    # read in road_network_attractions.csv
    attractions = pd.read_csv("road_network_attractions.csv")
    roadtrip_narrative = ""
    for city in cities:
        # Filter attractions for the current city
        found = attractions[attractions["Loc or Edge Label"] == city]
        city_attractions = found["Attraction Label"].tolist()
        description = found["Description"].tolist()
        if city_attractions:
            roadtrip_narrative += f"{city}: " + ", ".join(city_attractions) + ", ".join(description) + ".\n"
        else:
            roadtrip_narrative += f"{city}: [pick random attraction from the city and then describe using chain of thought].\n"

    roadtrip_narrative += "Finish out the roadtrip with saying welcome home to " + cities[0] + "! Where you first started your journey." 
    print(roadtrip_narrative)
    give_narrative(roadtrip_narrative)
    # roadtrip = """
    # Ruby Falls, Chattanooga TN: Today, Ruby Falls welcomes guests to Lookout Mountain from around the world to enjoy underground cave adventures, spectacular views of the Cumberland Plateau, soaring zip lines and award-winning special events! Activities: Hiking, Landmark, Environment, Nature.
    # Centennial Park, Nashville TN: Centennial Park is one of Nashville's premier parks. Located on West End and 25th Avenue North, the 132-acre features the iconic Parthenon, a one-mile walking trail, Lake Watauga, the Centennial Art Center, historical monuments, an arts activity center, a beautiful sunken garden, a band shell, an events shelter, sand volleyball courts, dog park, and an exercise trail. Thousands of people visit the park each year to visit the museum, see exhibits, attend festivals, and just enjoy the beauty of the park. Activities: Walking, Landmark, Environment, Nature.
    # Vanderbilt University, Nashville TN: Vanderbilt is a private research university in Nashville, Tennessee. It offers more than 70 undergraduate majors and a full range of graduate and professional degrees across 10 schools and colleges, all on a beautiful campus—an accredited arboretum—complete with athletic facilities and state-of-the-art laboratories. Activities: Building, University, Walking.
    # """

    # give_narrative(roadtrip)

