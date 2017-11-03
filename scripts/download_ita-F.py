import requests
from lxml import html
from bs4 import BeautifulSoup

names = ["Aba"]
url = "https://it.wikipedia.org/w/index.php?title=Categoria:Prenomi_italiani_femminili&pagefrom=%s+%%28nome%%29#mw-pages"

next_first = names[0]

while(True):

    page = requests.get(url % (next_first))

    soup = BeautifulSoup(page.content, 'html.parser')

    if not soup.find('div', 'mw-category'):
        break

    tags = soup.find('div', 'mw-category').find_all('a')

    names.pop()
    for name in tags:
        names.append(name.text.replace(' (nome)', ''))

    next_first = names[-1]

outfile = open('data/ita_F.csv', 'w')

for item in names:
    outfile.write("%s\n" % item)

    print(item)