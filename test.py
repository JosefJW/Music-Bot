import wikipedia

def get_wikipedia_article(song_title):
    try:
        page = wikipedia.page(song_title)
        return page.content
    except Exception as e:
        return None
    
print(get_wikipedia_article("Out of the frying pan "))