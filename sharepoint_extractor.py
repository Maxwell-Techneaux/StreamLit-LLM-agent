import requests
import re
import os, json 
import string
from dotenv import load_dotenv
import msal
from datetime import datetime, timezone
from playwright.sync_api import sync_playwright
from urllib.parse import urlparse, urlunparse, urljoin, quote
from bs4 import BeautifulSoup
from LlamaFolder.document_parser import DocumentParser
from langchain_core.documents import Document 
import base64

# access .env
load_dotenv()

# txt file with last sharepoint access date + time
INDEX_PATH = r"C:\Users\maxwell.boutte\Techneaux Interns\LlamaFolder\index_list.txt"
# define the image dictionary path
JSON_PATH = r"C:\Users\maxwell.boutte\Techneaux Interns\imageByte.json"

class EnvironVariables:
    # illegal activity that no one can know about
    TENANT_ID = os.getenv('TENANT_ID')
    CLIENT_ID = os.getenv('O365_CLIENT_ID')
    API_BASE = os.getenv('GRAPH_API_BASE') 
    CLIENT_SECRET = os.getenv('O365_CLIENT_SECRET')
    site_url = os.getenv('SHAREPOINT_SITE') 
    LAST_INDEX_FILE_PATH = INDEX_PATH


# parse url and clean 
def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path.rstrip("/").lower(),
        '', '', ''
    ))


# parse through HTML to find glossary terms and their definitions 
def extract_gloss_html(html: str):
    
    scraped_url = "https://softwaredocs.weatherford.com/cygnet/94/Content/Topics/CygNet%20Software/Glossary%20of%20Terms.htm"
    base_url = scraped_url # no preformat necessary
    
    # load soup for html reading
    soup = BeautifulSoup(html, 'html.parser') 
    

    # all definitions are in paragraphs, all term names are headers of these paragraphs
    all_elements = soup.find_all(['h3', 'p'])

    # paired lists for each term
    terms = [] # list of all term-def pairs
    current_term = None # most recently pulled term
    current_def = [] # lines for definition

    for elem in all_elements:
        # terms then paragraphs
        if elem.name == 'h3':
            # Save the previous term
            if current_term:
                terms.append((current_term, ' '.join(current_def).strip()))
            current_term = elem.get_text(strip=True)
            current_def = []

        # definition lives here    
        elif elem.name == 'p':
            # find attributes of the paragraph that contain a valid hyperlink
            for a_tag in elem.find_all('a', href=True):
                href = a_tag['href'] # Makes links fully clickable
                parsed = urlparse(href)

                # if the link is a redirect to the same page (scroll-down to keyword), drop the extension into the main URL
                if href.startswith('#'):
                    a_tag['href'] = base_url.split('#')[0] + href

                # otherwise, build it from the base url
                elif not parsed.netloc:
                    abs_url = urljoin(base_url, href)
                    parsed_abs = urlparse(abs_url)
                    encoded_path = quote(parsed_abs.path)
                    new_url = urlunparse((
                        parsed_abs.scheme,
                        parsed_abs.netloc,
                        encoded_path,
                        parsed_abs.params,
                        parsed_abs.query,
                        parsed_abs.fragment
                    ))
                    a_tag['href'] = new_url
                    #a_tag['href'] = urljoin(base_url, href)
                else:
                    encoded_path = quote(parsed.path)
                    new_url = urlunparse((
                        parsed.scheme,
                        parsed.netloc,
                        encoded_path,
                        parsed.params,
                        parsed.query,
                        parsed.fragment
                    ))
                    a_tag['href'] = new_url
            # basic text for definition, including hyperlink attribute where available.   
            current_def.append(str(elem))

    # Add the last term to end of terms
    if current_term:
        terms.append((current_term, ' '.join(current_def).strip()))

    # feed back terms list for document creation
    return terms


# parse through HTML to find steps and images for those steps
def extract_clean_html(html: str) -> tuple[str, list[str]]:
    

    soup = BeautifulSoup(html, "html.parser")
    steps = []

    # Find the <ol> with specific class and style
    ol_tags = soup.find_all("ol", class_="customListStyle", style="list-style-type:decimal;")
    
    # search for overarching list of instructions first, ignoring .aspx pages without them such as the Home.aspx redirect
    if not ol_tags:
        print("❌ Could not find the <ol> with the specified class and style.")
        return ("", [])
    
    sibling = None

    print(f"✅ Found {len(ol_tags)} <ol> elements. Parsing all...")

    for ol_tag in ol_tags:
    #search for adjacent lines  
        sibling = ol_tag.find_next_sibling()

    # Loop through direct <li> children
        for step_number, li in enumerate(ol_tag.find_all("li", recursive=False), start=1):
            
            # get exact text from the line
            p_tag = li.find("p")
            main_text = p_tag.get_text(separator=" ", strip=True) if p_tag else li.get_text(separator=" ", strip=True)

            # Look for sub <ol> inside this line
            sub_ol = li.find("ol")
            if sub_ol:
                letters = string.ascii_lowercase
                sub_texts = []
                # instances of lines within the step are prepended with consecutive letters 
                for i, sub_li in enumerate(sub_ol.find_all("li", recursive=False)):
                    sub_text = sub_li.get_text(separator=" ", strip=True)
                    letter = letters[i] if i < len(letters) else '?'
                    sub_texts.append(f"{letter}. {sub_text}")
                # Append substeps separated by newlines
                text = main_text + "\n" + "\n".join(sub_texts)
            else:
                # if no sub-outline, store only the step line text
                text = main_text
            # generally no image
            image_url = None
        
            # find the next line that is a division of class imagePlugin
            img_divs = li.find_all("div", class_="imagePlugin")
            # for all steps, search for nearest next image url(s) in HTML
            if img_divs:
                for div in img_divs:
                    url = div.get("data-imageurl")
                    if url:
                        # strip escape sequence and restructure URL extension into accessible link
                        url = url.replace("&amp;", "&")
                        image_url = urljoin("https://"+EnvironVariables.site_url+"/", url)
            
            # some images are not direct children of the lines, so we iterate on next lines until we find an image or a new step
            if image_url is None:
                # Move sibling pointer forward until we find next imagePlugin or reach None
                while sibling and not (sibling.name == "div" and "imagePlugin" in sibling.get("class", [])):
                    # try again
                    sibling = sibling.find_next_sibling()
                if sibling and sibling.name == "div" and "imagePlugin" in sibling.get("class", []):
                    # repeat "if img_divs:" logic if a picture is found  
                    url = sibling.get("data-imageurl")
                    if url:
                        url = url.replace("&amp;", "&")
                        # make pretty 
                        image_url = urljoin("https://" + EnvironVariables.site_url + "/", url)
                    
                    # Move sibling pointer ahead for next iteration
                    sibling = sibling.find_next_sibling()

        # drop current step and its image(s) into the list of steps
            steps.append((f"{step_number}. {text}", image_url))

    # return the list of step/image pairs from this .aspx page
    return steps


# ask for permission, not forgiveness <3
def token_getter(tenant_id, client_id, client_secret, scope):
    # beg microsoft for a token PLEASE
    authority = f"https://login.microsoftonline.com/{tenant_id}"
    # keep it a secret 
    app = msal.ConfidentialClientApplication(client_id, client_credential=client_secret, authority=authority)
    result = app.acquire_token_for_client(scopes = [scope])
    if "access_token" in result: # give token
        return result["access_token"]
    else:   # lol no token
        raise Exception(f"No token: {result.get('error_description')}")


# read txt file of current date to only pull aspx pages that have been updated since that date
def get_last_index_time():
    """
    Retrieve the timestamp of the last successful indexing run from disk.
    Returns 1970-01-01 if not available (i.e., first run).
    """
    if os.path.exists(EnvironVariables.LAST_INDEX_FILE_PATH):
        with open(EnvironVariables.LAST_INDEX_FILE_PATH, "r") as f:
            return datetime.fromisoformat(f.read().strip())
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


# update txt file with current time after pulling pages
def update_last_index_time():
    if os.path.exists(EnvironVariables.LAST_INDEX_FILE_PATH):
        with open(EnvironVariables.LAST_INDEX_FILE_PATH, "w") as f:
            f.write(datetime.now(timezone.utc).isoformat())


# pull pages with permission
def page_getter(token, site, time):

    # keys to the kingdom 
    headers = {"Authorization": f"Bearer {token}"}
    docs = []
    site_api_url = f"https://graph.microsoft.com/v1.0/sites/{site}:/sites/B&MBU"
    site_id = requests.get(site_api_url,headers=headers).json().get("id")
    page_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/pages/microsoft.graph.sitePage"
    
    # lets make sure we can get in 
    try:
        pages_response = requests.get(page_url,headers=headers)
    except requests.HTTPError as e:
        print(f"[WARN] Failed to fetch or parse {url}: {e}")
  
    # *hacker voice* "We're in."
    with sync_playwright() as p:
        # give playwright a fake browser
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        
        # run playwright and ask for login stuff
        context.new_page().goto("https://techneaux1.sharepoint.com/sites/B&MBU")
        input("Log into SharePoint in browser, then press Enter to continue...")
        context.storage_state(path="auth_state.json")

        # get all associated pages
        pages = pages_response.json().get("value", [])
        
        
        for page_meta in pages:
            url = page_meta.get("webUrl")
            modified = datetime.fromisoformat(page_meta.get("lastModifiedDateTime", "1970-01-01")) #.replace("Z", "+00:00")
            # Skip old or malformed entries, only pulling recently modified pages that contain contents
            if not url or modified <= time:
                continue

            try:
                # attempt to access each page
                tab = context.new_page()
                tab.goto(url, timeout=60000)
                # save html 
                html = tab.content()
                # close this page
                tab.close()

                # get the steps for this page
                step_data = extract_clean_html(html)
                # words go here
                clean_text_lines = []
                # links to images go here
                image_urls = []
                # images data go here
                image_bytes = []
                
                #
                for step_text, image_url in step_data:
                    # each page's source and modified date are added as metadata along with the list of image URLs
                    metadata = {
                        "source": normalize_url(url).replace(" ", "%20"),
                        "last_modified": modified.isoformat(),
                        "images": [image_url] if image_url else [] 
                    }
                    
                    # for steps that have an image, append the full URL the text for streamlit to split later
                    if image_url:
                        try:
                            # make sure image loads 
                            response = context.request.get(image_url)
                            if response.ok:
                                # extract raw image file 
                                encoded = base64.b64encode(response.body()).decode("utf-8")
                                # add images to the list of images. These lists constitute the image dictionary.
                                image_urls.append(image_url)
                                image_bytes.append(encoded)
                                # include markdown indicator for StreamLit
                                step_text += f" ![Image]({image_url})"
                        except Exception as e:
                            print(f"[WARN] Failed to fetch image {image_url}: {e}")
                    
                    # stick the full text from each step, including a URL if necessary, into the list of steps
                    clean_text_lines.append(step_text)

                # make one big ol block of steps to save as a document
                clean_text = "\n".join(clean_text_lines)

                # update the image dictionary by linking the image lists
                image_dict = {image_urls[i]: image_bytes[i] for i in range(len(image_urls))}
                
                # load the existing image dictionary, if it exists
                if os.path.exists(JSON_PATH):
                    with open(JSON_PATH, "r") as f:
                        existing_image_dict = json.load(f)
                else:
                    existing_image_dict = {}

                # update the directory with new images (total refresh, no duplicate entries)
                existing_image_dict.update(image_dict)

                # write the updated dictionary to the file
                with open(JSON_PATH, "w") as f:
                    json.dump(existing_image_dict, f, indent=2)

                # Create LangChain document for each page 
                docs.append(Document(
                    page_content=clean_text,
                    metadata=metadata
                ))
            except Exception as e:
                print(f"[WARN] Failed to fetch or parse {url}: {e}")
        
        # done with SharePoint
        context.close()
        browser.close()
        

        # relaunch the browser under new context for CygNet definitions
        glossary_browser = p.chromium.launch(headless=True) 
        glossary_context = glossary_browser.new_context()

        # go to CygNet glossary
        try:
            glossary_page = glossary_context.new_page()
            glossary_page.goto("https://softwaredocs.weatherford.com/cygnet/94/Content/Topics/CygNet%20Software/Glossary%20of%20Terms.htm", timeout=60000)
            glossary_html = glossary_page.content()
            glossary_page.close()

            # rip glossary terms and definitions
            glossary_items = extract_gloss_html(glossary_html)
            
            # tag each source with the releveant term and append as singular documents
            for term, definition in glossary_items:
                
                docs.append(Document(
                    page_content=f"{term}:\n{definition}",
                    metadata={"source": f"Cygnet Glossary of Terms: {term}",
                              "type": "glossary",
                              "term": term}
                ))
            
### MAYBE WE WANT THIS MEASUREMENT INFO VERY USEFUL###

            # glossary_page = glossary_context.new_page()
            # glossary_page.goto("https://softwaredocs.weatherford.com/cygnet/94/Content/Topics/CygNet%20Measurement/CygNet%20Measurement%20Concepts.htm", timeout=60000)
            # glossary_html = glossary_page.content()
            # glossary_page.close()

            # # rip glossary terms and definitions
            # glossary_items = extract_gloss_html(glossary_html)
            
            # # tag each source with the releveant term and append as singular documents
            # for term, definition in glossary_items:
                
            #     docs.append(Document(
            #         page_content=f"{term}:\n{definition}",
            #         metadata={"source": f"Cygnet Glossary of Terms: {term}",
            #                   "type": "glossary",
            #                   "term": term}
            #     ))

### NOTE ###
### try to preserve Links in definition ###

        except Exception as e:
            print(f"[ERROR] Failed to fetch glossary: {e}")
        finally:
            # stop the cygnet presses! 
            glossary_context.close()
            glossary_browser.close()    


    # remember this day
    update_last_index_time()

    # Loop through your documents and print token length
    for i, doc in enumerate(docs):
        tokens = re.findall(r"\w+|[^\w\s]", doc.page_content)
        print(f"token length{i}: ", len(tokens))

    # return the full list of training documents
    return docs


if __name__ == "__main__":
    # instance of .env variables 
    envar = EnvironVariables()

    # access the databasing/file functions
    parser = DocumentParser(
            pdf_path=r"C:\Users\maxwell.boutte\Techneaux Interns\PDFs",
            vector_path="./sql_chroma_db",
            memory_vector_path="./memory_chroma_db"
        )

    # stake your claim on sharepoint
    tok = token_getter(
        tenant_id = envar.TENANT_ID, 
        client_id = envar.CLIENT_ID, 
        client_secret = envar.CLIENT_SECRET, 
        scope = "https://graph.microsoft.com/.default")
    # 5,000 characters of pure unbridled permission
    print(tok)
    
    # when was the last time we met?
    last_index = get_last_index_time()
    print("last index: " + str(last_index))

    # grab documents updated after last_index
    documents_to_save = page_getter(tok, envar.site_url, last_index)

    print(f"Total documents: {len(documents_to_save)}")
    print(f"Documents with non-empty content: {sum(1 for d in documents_to_save if d.page_content and d.page_content.strip())}")
  
    parser.update_chroma_index(documents_to_save, r"C:\Users\maxwell.boutte\Techneaux Interns\sql_chroma_db", parser)