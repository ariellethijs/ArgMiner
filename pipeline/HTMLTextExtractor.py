import re

from bs4 import BeautifulSoup
import PyPDF2
import requests

class HTMLTextExtractor:
    def __init__(self, url):
        self.url = url

    def fetch_html(self):
        HEADERS = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36'}
        response = requests.get(self.url, headers=HEADERS)
        if response.status_code == 200:
            return response.text
        else:
            print("Failed to fetch HTML:", response.status_code)
            return None

    def extract_text(self):
        html_content = self.fetch_html()
         #deal with case that paragraph no. in <li> tag
        if html_content:
            soup = BeautifulSoup(html_content, 'html.parser')
            for li in soup.find_all('li'):
                value = li.get('value')
                if value:
                    li_text = li.get_text(strip=True)
                    li.replace_with(f"{value}. {li_text}")


            text = soup.get_text()
            lines = text.split('\n')
            index_start=-1
            index_end = -1
            lines = [line.strip() for line in lines if line.strip()]
            citation = lines[0]

            # adjusted regex to handle civil, criminal, and ex parte cases
            match = re.match(
                r'(.+?),?\s*(R v\.|v\.|v|Ex Parte)\s+(.+?)\s+(\[\d{4}\]\s+(UKSC|UKHL)\s+\d+)\s*(\(\d{1,2}[a-z]{2}\s+[A-Z][a-z]+\s+\d{4}\))?',
                citation
            )
            if match:
                case_type = match.group(2)
                if case_type == "R v.":  # Criminal case
                    appellant = match.group(1)
                    respondent = "R"
                elif case_type.lower() == "ex parte":  # Ex Parte case
                    appellant = match.group(1)
                    respondent = f"Ex Parte {match.group(3)}"
                else:  # Civil case
                    appellant = match.group(1)
                    respondent = match.group(3)
                cite = match.group(4)

            # print(citation)
            # match = re.match(r'(.+?) (v|v.) (.+?) (\[(\d{4})\] (UKSC|UKHL) \d+)', citation)
            # if match:
            #     respondent = match.group(1)
            #     appellant = match.group(3)
            #     cite = match.group(4)

            for i, line in enumerate(lines):
                match = re.match(r'\d+\.', line.strip())
                if match:
                    index_one = i
                    break
            for i, line in enumerate(lines):
                if 'Copyright' in line:
                    index_end = i
                    break
            for i, line in enumerate(lines[:index_one]):
                if line.upper().startswith("LORD"):
                    index_start = i
            if index_start != -1 & index_end != -1:
                text = '\n'.join(lines[index_start:index_end])
                text = cite + '\n' + text


            match = re.search(r'UK\w{2}/\d+/\d+', self.url)
            if match:
                file_path = "data/UKHL_txt/" + re.sub(r'[^\w\s-]', '', match.group()) + ".txt"

            else:
                print("No valid pattern found in the URL.")

            '''respondent_paragraphs = soup.find_all(string=re.compile(r'Respondent', re.IGNORECASE))
            if respondent_paragraphs:
                respondent = re.sub(r'\(RESPONDENT.*', '', respondent_paragraphs[0].get_text(), flags=re.IGNORECASE)
                if respondent.strip() == '':
                    respondent_paragraphs = soup.find_all('p', string=re.compile(r'Respondent', re.IGNORECASE))
                    previous_sibling = respondent_paragraphs[0].find_previous()
                    if previous_sibling is not None:
                        respondent = previous_sibling.get_text()
                        print('before respondent: ', respondent)
                    else:
                        print('No previous sibling found for the RESPONDENT paragraph.')
            else:
                if soup.find_all(string=re.compile(r'REGINA', re.IGNORECASE)):
                    respondent = 'REGINA'
                else:
                    respondent = 'Empty'


            appellant_paragraphs = soup.find_all(string=re.compile(r'APPELLANT', re.IGNORECASE))
            if appellant_paragraphs:
                appellant = re.sub(r'\(APPELLANT.*', '', appellant_paragraphs[0].get_text(), flags=re.IGNORECASE)
                if appellant.strip() == '':
                    appellant_paragraphs = soup.find_all('p', string=re.compile(r'APPELLANT', re.IGNORECASE))
                    previous_sibling = appellant_paragraphs[0].find_previous()
                    if previous_sibling is not None:
                        appellant = previous_sibling.get_text()
                        print('before appellant: ', appellant)
                    else:
                        print('No previous sibling found for the appellant paragraph.')
            else:
                appellant = 'Empty' '''

            text = text + '\n' + respondent + '\n' + appellant + '\n'

            with open(file_path, "w", encoding="utf-8") as file:

                file.write(text)
            print("Text saved to", file_path)
            return file_path
        else:
            return None

'''Example usage:
url = "https://www.bailii.org/uk/cases/UKHL/2001/2.html"

HTMLTextExtractor(url).extract_text()'''