import requests
from lxml import html
from bs4 import BeautifulSoup

names = ["Aarav"]
url = "https://en.wiktionary.org/w/index.php?title=Category:English_male_given_names&pagefrom=%s#mw-pages"

next_first = names[0]

while(True):

    page = requests.get(url % (next_first))

    soup = BeautifulSoup(page.content, 'html.parser')

    tag = soup.find('div',id='mw-pages')

    tags = tag.find_all('a')

    names.pop()

    for name in tags:
        if name.text != 'previous page' and name.text != 'next page' :
            names.append(name.text.split(' (')[0])

    if next_first == names[-1]:
        break    

    next_first = names[-1]

    print('next: '+next_first)

outfile = open('data/eng_M.csv', 'w')

for item in names:
  outfile.write("%s\n" % item)
