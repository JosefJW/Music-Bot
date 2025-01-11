import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import time
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

# Step 1: Load Data from SQL Database in Chunks
def load_data_in_batches(db_path, batch_size=1000):
    """Load data from the SQL database in batches."""
    conn = sqlite3.connect(db_path)
    query = "SELECT name, article FROM songs"
    offset = 0
    while True:
        batch_query = f"{query} LIMIT {batch_size} OFFSET {offset}"
        batch_data = pd.read_sql_query(batch_query, conn)
        if batch_data.empty:
            break
        yield batch_data  # This will yield a batch of songs as a DataFrame
        offset += batch_size
    conn.close()

def preprocess_data(data):
    """
    Tokenize the data, remove stop words, punctuation, apply stemming and lemmatization.
    """
    tokenized_data = []
    
    print("Tokenizing data...")
    start = time.time()
    for index, row in data.iterrows():
        title, article = row['name'], row['article']
        tokenized_article = nltk.word_tokenize(article)
        tokenized_data.append([title, tokenized_article])
        
        if index % 100 == 0:
            print(f"Tokenization has been running for: {time.time() - start} seconds")
            print(f"{index / len(data) * 100}% complete")
            print()
            
    # Process each song's title and article
    processed_data = []
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    print("Filtering, stemming, and lemmatizing data")
    start = time.time()
    for index, (title, article_tokens) in enumerate(tokenized_data):
        # Filter out stop words and punctuation
        filtered_article = [word.lower() for word in article_tokens if word.lower() not in stop_words and word not in string.punctuation]
        
        # Apply stemming and lemmatization
        stemmed_article = [stemmer.stem(word) for word in filtered_article]
        lemmatized_article = [lemmatizer.lemmatize(word, wordnet.VERB) for word in stemmed_article]
        
        processed_data.append([title, lemmatized_article])
        
        if index % 100 == 0:
            print(f"Processing for {time.time() - start} seconds")
            print(f"{index / len(tokenized_data) * 100}% complete")
            print()

    print("Data preprocessed!")
    return processed_data

def preprocess_article(article):
    """
    Tokenize, remove stop words, punctuation, apply stemming and lemmatization to a single article.
    """
    # Tokenize the article
    tokenized_article = nltk.word_tokenize(article)

    # Filter out stop words and punctuation
    filtered_article = [word.lower() for word in tokenized_article if word.lower() not in stop_words and word not in string.punctuation]

    # Apply stemming and lemmatization
    stemmer = PorterStemmer()
    stemmed_article = [stemmer.stem(word) for word in filtered_article]
    lemmatizer = WordNetLemmatizer()
    lemmatized_article = [lemmatizer.lemmatize(word, wordnet.VERB) for word in stemmed_article]

    return lemmatized_article

def vectorize_data(data, vectorizer):    
    titles = [item[0] for item in data]
    article_texts = [" ".join(item[1]) for item in data]
    
    article_vectors = vectorizer.fit_transform(article_texts)
    
    article_title_mapping = {titles[i]: article_vectors[i] for i in range(len(titles))}
    return article_title_mapping

def recommend_songs(user_article, article_title_mapping, vectorizer, top_n=10000):
    # Convert the user's article into a vector
    user_article = " ".join(user_article)
    user_vector = vectorizer.transform([user_article])
    
    # Compute cosine similarities between the user's vector and all article vectors
    similarities = cosine_similarity(user_vector, np.vstack([vec.toarray() for vec in article_title_mapping.values()]))
    
    # Flatten the similarity matrix to make it easier to work with
    similarities = similarities.flatten()
    
    # Get the indices of the top N most similar songs
    top_n_indices = similarities.argsort()[-top_n:][::-1]
    
    # Get the song titles and their similarity scores
    recommended_songs_with_scores = [(list(article_title_mapping.keys())[index], similarities[index]) for index in top_n_indices]
    
    # Create a dictionary to hold the songs grouped by similarity score
    grouped_by_similarity = {}
    
    for song, score in recommended_songs_with_scores:
        if score not in grouped_by_similarity:
            grouped_by_similarity[score] = []
        grouped_by_similarity[score].append((song, score))
    
    # Now, for each group of songs with the same similarity score, select the one with the longest title
    final_recommendations = []
    
    for score, songs in grouped_by_similarity.items():
        # Get the song with the longest title in this group
        longest_title_song = max(songs, key=lambda x: len(x[0]))
        final_recommendations.append(longest_title_song)
    
    # Sort the final recommendations by similarity score in descending order
    final_recommendations.sort(key=lambda x: x[1], reverse=True)
    
    return final_recommendations

db_path = "songs.db"
vectorizer = TfidfVectorizer(stop_words=None)

my_article = """


Wikipedia The Free Encyclopedia

    Donate
    Create account
    Log in

Contents
(Top)
Background and composition

Release and commercial performance

    Initial release
    Resurgence

Critical reception

Awards and nominations

Usage in media

Personnel

Charts

        Weekly charts
        Monthly charts
        Year-end charts
    Certifications
    Release history
    See also
    References

Cruel Summer (Taylor Swift song)

    Article
    Talk

    Read
    Edit
    View history

Tools

Appearance
Text

    Small
    Standard
    Large

Width

    Standard
    Wide

Color (beta)

    Automatic
    Light
    Dark

From Wikipedia, the free encyclopedia
"Cruel Summer"
The single cover sees Taylor Swift lying on a picnic rug on grass, wearing sunglasses
Single by Taylor Swift
from the album Lover
Released	June 13, 2023
Recorded	2019
Studio	

    Electric Lady (New York City)
    Conway Recording (Los Angeles)

Genre	

    Synth-pop electropop industrial pop

Length	2:58
Label	Republic
Songwriter(s)	

    Taylor Swift Jack Antonoff Annie Clark

Producer(s)	

    Taylor Swift Jack Antonoff

Taylor Swift singles chronology
"Karma"
(2023) 	"Cruel Summer"
(2023) 	"'Slut!'"
(2023)
Audio video
"Cruel Summer" on YouTube

"Cruel Summer" is a song by the American singer-songwriter Taylor Swift from her seventh studio album, Lover (2019). Swift and Jack Antonoff produced the song, and they wrote it with St. Vincent. "Cruel Summer" is a synth-pop, industrial pop, and electropop song composed of synths, wobbling beats, and vocoder-manipulated vocals. The lyrics are about an intense romance during a painful summer.

When it was first released as an album track on Lover, music critics praised "Cruel Summer" for its melodic composition and catchy sound, specifically highlighting the hook and bridge. Many deemed it a highlight on Lover and one of Swift's best songs. "Cruel Summer" debuted in the top 30 of various singles charts in 2019 and became a fan favorite over time, prompting fans and publications to question why Swift did not release the track as a single. After being included in the set list of Swift's sixth concert tour, the Eras Tour, in 2023, "Cruel Summer" became viral on social media, leading Republic Records to release it as a single on June 13, 2023.

In the United States, the single peaked atop the Billboard Hot 100 and helped Swift become the solo artist with the most number-one songs on the Pop Airplay and Adult Pop Airplay radio charts. Elsewhere, it topped the Billboard Global 200 and the singles charts in Australia, Canada, the Philippines, and Singapore. It was the seventh-most-streamed song globally of 2023 and was placed on the 2024 revision of Rolling Stone's 500 Greatest Songs of All Time.
Background and composition
St Vincent wearing red, singing onto a mic
St. Vincent co-wrote and played guitar on "Cruel Summer".

Taylor Swift described her seventh studio album, Lover, as a "love letter to love" itself with all the feelings evoked by it.[1] The album was released on August 23, 2019, via Republic Records.[2] Lover consists of 18 tracks, and "Cruel Summer" is track number two.[3] According to Swift, the track is about an uncertain summer romance with elements of pain and desperation in it.[4] The relationship in question is "where you're yearning for something that you don't quite have yet, it's just right there, and you just can't reach it".[4]
"Cruel Summer"
Duration: 18 seconds.0:18
A sample of the hook in "Cruel Summer", which has been described as "infectious" and "perfect" by many music critics.
Problems playing this file? See media help.

"Cruel Summer" is predominantly a synth-pop song.[5][6][7] Critics described its production as melancholic[8] or dreamy.[9] Mikael Wood of the Los Angeles Times categorized the song as industrial pop,[10] and Ludovic Hunter-Tilney of the Financial Times dubbed it electropop.[11][12] It has a "ranting" bridge underscored by skittering synths,[13][14][15][16] distorted vocals[5] manipulated by a vocoder,[17] and a hook that consists of a long, high, fluctuating "ooooh".[18] The song has a fast tempo of 170 beats per minute with a time signature of 4
4. It is played in the key of A major and follows a chord progression of A–C♯m–F♯m–D.[19][20] Swift vocals range from A2 to E5. "Cruel Summer" was written by Swift, Jack Antonoff and St. Vincent,[21] with a "burbling" production from Swift and Antonoff;[22] St. Vincent also took part in the production of the song, playing the guitar.[23] Lyrically, the song is about "the agony and ecstasy of an anxious summer romance".[24] David Penn of Hit Songs Deconstructed opined, the song's vocals, instrumentation and lyrics work "in tandem to create a unified expression, a combination known as prosody."[25]

It portrays the challenges faced by pop stars in the public spotlight.[13] The vulnerability of the song's lyrics has drawn comparisons to "Delicate", the fifth track on Swift's 2017 album Reputation.[13] Billboard's Heran Mamo opined that the song's lyrics see Swift "wrestling with strong feelings", where they paint "the picture of an emotional night out".[26] Justin Styles of The Ringer wrote that the song tells a "more humanizing version" of Swift's "ill-fated period three years ago", adding that Swift sings about "falling in love with then-current boyfriend Joe Alwyn while her public life was in shambles".[22] Anna Gaca, writing for Pitchfork, called the song a "drama-free delight" with "magnetic pink glow".[27] The Spinoff pointed out that Swift's vocals in "Cruel Summer" are "most notable for the modern country cadence".[14]
Release and commercial performance
Initial release

"Cruel Summer" was released as the second track on Lover, on August 23, 2019, via Republic Records.[28] The track originally charted as an album cut within the top 30 in Singapore (8),[29] Malaysia (13),[30] Ireland (20),[31] New Zealand (20),[32] Australia (23),[33] the United Kingdom (27),[34] and Canada (28).[35] In the United States, the song debuted at number 29 on the Billboard Hot 100 dated September 7, 2019; it is one of the seven tracks from Lover to reach the top 40[36] and remained on the chart for two weeks.[37] The song became a widespread fan favorite over time[38][39] and critics and fans questioned Swift's decision over not having released "Cruel Summer" as a single.[40][41]
Resurgence
See also: Impact of the Eras Tour
Swift singing onstage dressed in a sparkling bodysuite
"Cruel Summer" resurged in popularity after Swift included it in the setlist of the Eras Tour in 2023.

Beginning March 2023, Swift embarked on the Eras Tour, her sixth headlining concert tour, as a tribute to all of her "musical eras".[42] The show begins with the Lover act, during which "Cruel Summer" is the second song performed.[41] Around this time, the song began to resurge in popularity and streaming after it became viral on social media.[40][43] In the U.S., the single re-entered the Billboard Hot 100 at number 49 on the chart dated June 3, 2023.[37] As a result, Republic Records released it as the fifth Lover single to US contemporary hit radio on June 13, 2023.[44] The song also impacted hot adult contemporary radio on June 26.[45] On June 17, at an Eras Tour show in Pittsburgh, Swift said she had intended to release "Cruel Summer" as a single in 2020 during the promotional cycle for Lover, but she abandoned the plan after the outbreak of the COVID-19 pandemic and moved forward with detouring her artistic direction and releasing her next album, Folklore.[46][47]

"Cruel Summer" became Swift's record-extending 41st song to reach the top 10 on the Billboard Hot 100 and the fourth Lover track to do so.[48] After the release of the Eras Tour's accompanying concert film, a live recording of the song and a remix by LP Giobbi were released as part of a streaming compilation, titled The Cruelest Summer, on October 18, 2023.[49] "Cruel Summer" topped the Billboard Hot 100, marking Lover's first and Swift's 10th number-one single.[50] It was replaced by, and in turn replaced, Swift's "Is It Over Now?" atop the Hot 100 for one week,[51] spending a total of four non-consecutive weeks at number one and making Swift the first female artist to succeed herself at the top spot twice and thrice.[52][53]

On US Billboard airplay charts in 2023, "Cruel Summer" became Swift's eighth number-one single on Radio Songs, where it reigned for 12 non-consecutive weeks surpassing "Blank Space" as her longest running number one.[54] It became her 12th number-one single on Pop Airplay, and her 11th number-one single on Adult Pop Airplay, making her the solo artist with the most chart toppers on the latter two charts.[55][56] The song also spent 10 weeks atop Pop Airplay and 23 weeks atop Adult Pop Airplay, becoming her longest-running number-one song on both[54] and the longest-running number-one song by a solo artist and female artist on the latter.[53][57] Jason Lipshutz of Billboard commented that the single's resurgent success "simply demonstrates Swift's current ubiquity, unprecedented in the modern music era".[58] In January 2024, the song topped the Adult Contemporary chart, marking Swift's ninth number-one single. As such, it made Swift the first artist in history to release six singles that topped the Adult Contemporary, Adult Pop Airplay and Pop Airplay charts individually, surpassing Adele (five).[57] Additionally, the song spent 34 weeks in the top ten of the Hot 100, becoming the first song by a solo female artist to do so.[59] In total, "Cruel Summer" spent 54 weeks on the Hot 100, becoming Swift's longest-charting Hot 100 hit, surpassing "Anti-Hero".[60]

Elsewhere, "Cruel Summer" reached new peaks in Australia (1),[61] Canada (1),[62] Singapore (1),[63] New Zealand (3),[64] Ireland (4),[65] Malaysia (6),[66] and Brazil (54) as well.[67] It peaked at number one in the Philippines[68] and entered the top 10 in Indonesia.[69] The song has received certifications from Denmark (gold),[70] Greece (gold),[71] Italy (platinum),[72] New Zealand (triple platinum),[64] Poland (double platinum),[73] Portugal (triple platinum),[74] and the UK (triple platinum).[75] It reached number one on the Billboard Global 200,[76] and was the sixth most-streamed song globally on Spotify in 2023.[77] The song was the seventh-biggest song of 2023 according to the International Federation of the Phonographic Industry (IFPI), with an equivalent of 1.39 billion global subscription streams.[78] Within the first half of 2024, according to Luminate Data, the single was the third-most-streamed song globally with 1.012 billion streams.[79]

St. Vincent described the resurgent success of "Cruel Summer" as "crazy": "I mean, I always thought in the context of that record, like, 'That should be a single, it’s a great song.' And I don't even think it was a single; it just was a fan favorite. And it's like the fans just decided: 'No, this is your hit song.' Which is so wild and so modern, you know."[80]
Critical reception

In the reviews of Lover, "Cruel Summer" received rave reviews from music critics, particularly for its production, bridge and hook. Jon Caramanica of The New York Times commended the "thick, ethereal" production and Swift's signature vocal motifs such as the "question-mark syllables" and the "hard-felt smears".[81] Mikael Wood of the Los Angeles Times proclaimed "Cruel Summer" to be the best song of Lover and said the bridge where Swift "shrieks about the devil might be the punkest thing you'll hear all year".[82] Alex Abad-Santos, writing for Vox, listed "Cruel Summer" as one of his top-three best Lover tracks, writing that the song is an "aquatic robot bop" featuring "wobbly" synths.[83] The Spinoff stated that Swift "absolutely pulls it off" and compared it to the Bananarama's 1984 song of the same name.[14] Writing for The Ringer, Justin Sayles praised the song as a "better rebuke of her personal drama than anything on her last album", and added that Swift "shakes off the bad vibes" with "Cruel Summer"; Sayles named it Swift's "most infectious song since that run of singles from 1989", and opined that song "sets the tone" for the "warmer, more inviting vibes" of Lover.[22] Also calling it "infectious", Nick Levine of NME termed the track as a "brilliant pop song".[84] Natalia Barr, writing for Consequence, highlighted Swift's vocal delivery in the song's bridge ("He looks up, grinning like a devil"), calling it "simultaneously funny, agonizing, and thrilling, and needs to be created into a viral YouTube loop immediately". Barr further labeled "Cruel Summer" as one of the "most perfect" pop songs of 2019.[85] "Cruel Summer" featured on year-end lists of the best songs of 2019 by Rolling Stone (4th)[86] and Billboard (10th).[87]

Retrospectively, "Cruel Summer" continued to receive high acclaim, and has been deemed the signature track of Lover. In a 2021 list ranking the best bridges of the 21st-century, Billboard placed "Cruel Summer" at number 11.[88] The song has ranked highly on critics' rankings of Swift's songs in her discography, appearing on such lists by Rob Sheffield of Rolling Stone (2021) at number 11 out of 229,[89] and Hannah Mylrea of NME (2020), number 6 out of 161.[90] Clash critics picked the song as one of Swift's 15 best, citing its "highly addictive" song structure.[91] In 2022, Exclaim!'s Alex Hudson and Megan LaPierre ranked it second on another list of the best 20 songs by Swift, praising how St. Vincent's artistic input complements Swift's.[92] Allaire Nuss of Entertainment Weekly described it as a "buzzer-beating, angst-wielding anthem".[93] Brittany Spanos of Rolling Stone wrote in 2023, "Swift flaunts a rock-star edge alongside a grand sense of romantic urgency" in "Cruel Summer", making it one of her best songs.[40] Billboard opined in 2023 that "Cruel Summer" is both a fan and a critics' favorite.[94] In 2024, Rolling Stone ranked the song at number 400 in their updated list of the 500 Greatest Songs of All Time.[95]
Awards and nominations
Year 	Award 	Category 	Result 	Ref.
2024 	iHeartRadio Music Awards 	Song of the Year 	Nominated 	[96]
Pop Song of the Year 	Nominated
TikTok Bop of the Year 	Won
BMI Awards 	Most Performed Song of the Year 	Won 	[97]
Billboard Music Awards 	Top Billboard Global 200 Song 	Nominated 	[98]
Top Billboard Global (Excl. U.S.) Song 	Nominated
Top Radio Song 	Nominated
Usage in media

    American singer-songwriter Olivia Rodrigo performed the song for MTV's Alone Together Jam Session in 2020, which Swift subsequently praised.[99]
    Rodrigo later stated that "Cruel Summer" partially inspired her 2021 single "Deja Vu", eventually crediting Swift, Antonoff, and St. Vincent as co-writers; it peaked at number three on the U.S. Hot 100.[100]
    It was featured in the first season of Amazon Prime Video series The Summer I Turned Pretty in June 2022.[101]
    It was one of Swift's songs used in a November 2023 episode of the American dance competition television show Dancing With the Stars, which was a tribute episode in honor of Swift; American television personality Ariana Madix and dance choreographer Pasha Pashkov performed a rumba to a rendition of the song.[102]
    Australian singer G Flip covered the song for Triple J's Like a Version in January 2024.[103]
    American singer-songwriter Teddy Swims covered the song for BBC Radio 1's Live Lounge segment in January 2024.[104]

Personnel

    Taylor Swift – vocals, songwriter, producer
    Jack Antonoff – producer, songwriter, programmer, recording engineer, drums, keyboards, vocoder
    St. Vincent – songwriter, guitar
    Michael Riddleberger – drums
    Serban Ghenea – mixer
    John Hanes – mix engineer
    John Rooney – assistant recording engineer
    Laura Sisk – recording engineer
    Jon Sher – assistant recording engineer

Charts
Weekly charts
2019 weekly chart performance for "Cruel Summer" Chart (2019) 	Peak
position
Australia (ARIA)[33] 	22
Canada (Canadian Hot 100)[35] 	28
Czech Republic (Singles Digitál Top 100)[105] 	84
Greece International (IFPI)[106] 	57
Ireland (IRMA)[107] 	20
Malaysia (RIM)[30] 	13
New Zealand (Recorded Music NZ)[108] 	20
Portugal (AFP)[109] 	94
Scotland (OCC)[110] 	70
Singapore (RIAS)[29] 	8
Slovakia (Singles Digitál Top 100)[111] 	100
Sweden Heatseeker (Sverigetopplistan)[112] 	10
UK Singles (OCC)[34] 	27
US Billboard Hot 100[113] 	29
2023–2024 weekly chart performance for "Cruel Summer" Chart (2023–2024) 	Peak
position
Argentina (Argentina Hot 100)[114] 	64
Australia (ARIA)[61] 	1
Austria (Ö3 Austria Top 40)[115] 	16
Belgium (Ultratop 50 Flanders)[116] 	46
Brazil (Brasil Hot 100)[117] 	54
Canada (Canadian Hot 100)[62] 	1
Canada AC (Billboard)[118] 	1
Canada CHR/Top 40 (Billboard)[119] 	1
Canada Hot AC (Billboard)[120] 	1
CIS Airplay (TopHit)[121] 	51
Croatia (HRT)[122] 	7
Czech Republic (Rádio – Top 100)[123] 	4
Czech Republic (Singles Digitál Top 100)[124] 	43
Denmark (Tracklisten)[125] 	31
Finland Airplay (Radiosoittolista)[126] 	14
France (SNEP)[127] 	52
French Airplay (SNEP)[128] 	9
Germany (GfK)[129] 	15
German Airplay (Official German Charts)[130] 	1
Global 200 (Billboard)[76] 	1
Greece International (IFPI)[131] 	11
Hong Kong (Billboard)[132] 	11
Iceland (Tónlistinn)[133] 	6
India International Singles (IMI)[134] 	7
Indonesia (Billboard)[69] 	3
Ireland (IRMA)[135] 	4
Israel (Media Forest)[136] 	6
Italy (FIMI)[137] 	85
Japan (Japan Hot 100)[138] 	48
Japan Combined Singles (Oricon)[139] 	44
Latvia (EHR)[140] 	2
Latvia (LAIPA)[141] 	9
Latvian Airplay (LAIPA)[142] 	1
Lithuania (AGATA)[143] 	30
Luxembourg (Billboard)[144] 	17
Malaysia (Billboard)[145] 	3
Malaysia International (RIM)[146] 	2
MENA (IFPI)[147] 	19
Netherlands (Dutch Top 40)[148] 	7
Netherlands (Global Top 40)[149] 	4
Netherlands (Single Top 100)[150] 	11
New Zealand (Recorded Music NZ)[151] 	3
Nigeria (TurnTable Top 100)[152] 	59
Norway (VG-lista)[153] 	18
Panama (Monitor Latino)[154] 	10
Panama (PRODUCE)[155] 	10
Paraguay (Monitor Latino)[156] 	14
Philippines (Billboard)[68] 	1
Poland (Polish Airplay Top 100)[157] 	8
Poland (Polish Streaming Top 100)[158] 	58
Portugal (AFP)[159] 	17
Singapore (RIAS)[160] 	1
Slovakia (Rádio Top 100)[161] 	4
Slovakia (Singles Digitál Top 100)[162] 	38
South African Airplay (TOSAC)[163] 	2
South Korea (Circle)[164] 	72
Spain (PROMUSICAE)[165] 	63
Sweden (Sverigetopplistan)[166] 	13
Switzerland (Schweizer Hitparade)[167] 	9
Taiwan (Billboard)[168] 	11
UAE (IFPI)[169] 	3
UK Singles (OCC)[170] 	2
US Billboard Hot 100[171] 	1
US Adult Contemporary (Billboard)[172] 	1
US Adult Pop Airplay (Billboard)[173] 	1
US Dance/Mix Show Airplay (Billboard)[174] 	12
US Pop Airplay (Billboard)[175] 	1
Venezuela (Record Report)[176] 	41
Vietnam (Vietnam Hot 100)[177] 	12
	
Monthly charts
Monthly chart performance for "Cruel Summer" Chart (2023–2024) 	Peak
position
CIS (TopHit)[178] 	42
Paraguay (SGP)[179] 	58
South Korea (Circle)[180] 	72
Year-end charts
2023 year-end chart performance for "Cruel Summer" Chart (2023) 	Position
Australia (ARIA)[181] 	8
Austria (Ö3 Austria Top 40)[182] 	51
Canada (Canadian Hot 100)[183] 	13
Germany (GfK)[184] 	69
Global 200 (Billboard)[185] 	27
Global Singles (IFPI)[186] 	7
Iceland (Tónlistinn)[187] 	41
Netherlands (Dutch Top 40)[188] 	47
Netherlands (Single Top 100)[189] 	77
New Zealand (Recorded Music NZ)[190] 	19
Philippines (Philippines Hot 100)[191] 	8
Poland (Polish Airplay Top 100)[192] 	69
Sweden (Sverigetopplistan)[193] 	85
Switzerland (Schweizer Hitparade)[194] 	65
UK Singles (OCC)[195] 	11
US Billboard Hot 100[196] 	18
US Adult Contemporary (Billboard)[197] 	13
US Adult Pop Airplay (Billboard)[198] 	14
US Pop Airplay (Billboard)[199] 	11
2024 year-end chart performance for "Cruel Summer" Chart (2024) 	Position
Austria (Ö3 Austria Top 40)[200] 	63
Canada (Canadian Hot 100)[201] 	14
Estonia Airplay (TopHit)[202] 	57
France (SNEP)[203] 	95
Germany (GfK)[204] 	52
Global 200 (Billboard)[205] 	4
Iceland (Tónlistinn)[206] 	41
Netherlands (Dutch Top 40)[207] 	79
Netherlands (Single Top 100)[208] 	85
New Zealand (Recorded Music NZ)[209] 	16
Philippines (Philippines Hot 100)[210] 	16
South Korea (Circle)[211] 	109
Switzerland (Schweizer Hitparade)[212] 	35
UK Singles (OCC)[213] 	11
US Billboard Hot 100[214] 	12
US Adult Contemporary (Billboard)[215] 	2
US Adult Pop Airplay (Billboard)[216] 	3
US Pop Airplay (Billboard)[217] 	10
Venezuela Anglo (Record Report)[218] 	15

Certifications
Certifications for "Cruel Summer" Region 	Certification 	Certified units/sales
Australia (ARIA)[219] 	9× Platinum 	630,000‡
Austria (IFPI Austria)[220] 	Platinum 	30,000‡
Brazil (Pro-Música Brasil)[221] 	2× Platinum 	80,000‡
Denmark (IFPI Danmark)[70] 	Platinum 	90,000‡
France (SNEP)[222] 	Diamond 	333,333‡
Germany (BVMI)[223] 	Gold 	300,000‡
Italy (FIMI)[72] 	Platinum 	100,000‡
New Zealand (RMNZ)[224] 	5× Platinum 	150,000‡
Poland (ZPAV)[73] 	2× Platinum 	100,000‡
Portugal (AFP)[74] 	4× Platinum 	40,000‡
Spain (PROMUSICAE)[225] 	2× Platinum 	120,000‡
Switzerland (IFPI Switzerland)[226] 	Gold 	10,000‡
United Kingdom (BPI)[75] 	3× Platinum 	1,800,000‡
Streaming
Greece (IFPI Greece)[71] 	Platinum 	2,000,000†
Japan (RIAJ)[227] 	Gold 	50,000,000†
Sweden (GLF)[228] 	Platinum 	12,000,000†
South Korea (KMCA)[229] 	Platinum 	100,000,000†
Worldwide 	— 	1,319,000,000[230]

‡ Sales+streaming figures based on certification alone.
† Streaming-only figures based on certification alone.
Release history
Release dates and formats for "Cruel Summer" Region 	Date 	Format(s) 	Version 	Label(s) 	Ref.
United States 	June 13, 2023 	Contemporary hit radio 	Original 	Republic 	[44]
June 26, 2023 	Hot adult contemporary radio 	[45]
Italy 	September 15, 2023 	Radio airplay 	Island 	[231]
Various 	October 18, 2023 	

    Digital downloadstreaming

	

    LiveLP Giobbi remix

	Republic 	[49]
See also

    List of Billboard Hot 100 number ones of 2023
    List of Radio Songs number ones of the 2020s
    List of Billboard Pop Airplay number-one songs of 2023
    List of Billboard Adult Top 40 number-one songs of the 2020s
    List of Canadian Hot 100 number-one singles of 2023
    List of number-one songs of 2023 (Singapore)
    List of number-one songs of 2024 (Singapore)
    List of number-one singles of 2024 (Australia)

References

Aniftos, Rania (August 8, 2019). "Taylor Swift Calls Lover Album Her 'Love Letter to Love,' Details 2 Unreleased Tracks". Billboard. Retrieved July 25, 2024.
Coscarelli, Joe (August 23, 2019). "Taylor Swift Releases Lover the Old-Fashioned Way". The New York Times. Archived from the original on August 28, 2019. Retrieved July 25, 2024.
Lover (liner notes). Taylor Swift. Republic Records. 2019.
Mastrogiannis, Nicole (August 24, 2019). "Taylor Swift Shares Intimate Details of Lover Songs During Secret Session". iHeartMedia. Archived from the original on August 25, 2019. Retrieved April 30, 2020.
O'Connor, Roisin (August 23, 2019). "Taylor Swift: Her 100 Album Tracks – Ranked". The Independent. Archived from the original on December 3, 2019. Retrieved September 14, 2019.
Hudson, Alex; LaPierre, Megan (October 20, 2022). "Taylor Swift's 20 Best Songs Ranked". Exclaim!. Archived from the original on December 6, 2022. Retrieved December 6, 2022.
Sun, Curtis (October 25, 2022). "All 285 Songs Jack Antonoff Has Produced, Ranked: See The List". Consequence. Archived from the original on June 15, 2023. Retrieved May 23, 2023.
Zaleski, Annie (August 26, 2019). "Taylor Swift Is Done Proving Herself on the Resonant Lover". The A.V. Club. Archived from the original on August 26, 2019. Retrieved April 30, 2020.
"Taylor Swift Launches Eras Tour with Three-Hour, 44-Song Set". BBC News. March 18, 2023. Archived from the original on March 18, 2023. Retrieved May 23, 2023.
Wood, Mikael (August 25, 2019). "Taylor Swift's 'Lover': All 18 songs, ranked". Los Angeles Times. Archived from the original on September 14, 2019. Retrieved September 14, 2019.
Hunter-Tilney, Ludovic (August 23, 2019). "Taylor Swift: Lover – Cupid's arrow hits the bullseye". Financial Times. Archived from the original on August 24, 2019. Retrieved July 28, 2020.
Lee, Taila (January 26, 2023). "The Taylor Swift Essentials: 13 Songs That Display Her Storytelling Prowess And Genre-Bouncing Genius". The Recording Academy. Archived from the original on April 14, 2023. Retrieved April 14, 2023.
Bruner, Raisa (August 23, 2019). "Let's Discuss the Lyrics to Every Song on Taylor Swift's Lover". Time. Archived from the original on June 9, 2020. Retrieved April 30, 2020.
"The Spinoff reviews all 18 songs on Taylor Swift's Lover". The Spinoff. August 26, 2019. Archived from the original on August 3, 2020. Retrieved April 30, 2020.
Wood, Mikael (August 24, 2019). "Review: Taylor Swift's 'Lover' courts — gasp! — adults with grown-up emotional complexity". Los Angeles Times. Archived from the original on August 24, 2019. Retrieved April 30, 2020.
"Taylor Swift and Jack Antonoff's 20 Best Collaborations". Slant Magazine. November 6, 2022. Archived from the original on March 2, 2023. Retrieved May 23, 2023.
Willman, Chris (December 13, 2022). "Taylor Swift's 50 Best Songs, Ranked". Variety. Archived from the original on January 8, 2023. Retrieved May 23, 2023.
McCormick, Neil (August 23, 2019). "Taylor Swift, Lover, review: zippy, feminist electropop about young love – and watching rugby down the pub". The Daily Telegraph. ISSN 0307-1235. Archived from the original on August 23, 2019. Retrieved May 20, 2020.
"Cruel Summer Sheet Music". Musicnotes. August 30, 2019. Archived from the original on April 24, 2021. Retrieved May 4, 2020.
"Key & BPM/Tempo of Cruel Summer by Taylor Swift". Archived from the original on April 24, 2021. Retrieved May 5, 2020.
Aubrey, Elizabeth (August 22, 2019). "St Vincent has worked with Taylor Swift on a new song, 'Cruel Summer'". NME. Archived from the original on June 9, 2020. Retrieved April 30, 2020.
Sayles, Justin (August 23, 2019). "Taylor Swift Shakes Off the Bad Vibes With "Cruel Summer"". The Ringer. Archived from the original on December 13, 2019. Retrieved April 30, 2020.
Lipshutz, Jason (August 23, 2019). "Taylor Swift's 'Lover' Analysis: On The Free-Spirited Album, She Does What She Wants, When She Wants". Billboard. Archived from the original on February 23, 2020. Retrieved April 30, 2020.
Tousignant, Lauren (May 13, 2023). "Taylor Swift, Give Us the 'Cruel Summer' Music Video, You Coward". Jezebel. Archived from the original on June 15, 2023. Retrieved June 15, 2023.
Penn, David (October 18, 2023). "Taylor Swift's 'Cruel Summer' Deconstructed: Strategic Hooks, Intriguing Lyrics & an Emotional Connection". Billboard. Archived from the original on October 23, 2023. Retrieved October 24, 2023.
Mamo, Heran (September 6, 2019). "Taylor Swift's 'Cruel Summer' Lyrics". Billboard. Archived from the original on March 1, 2020. Retrieved April 30, 2020.
Gaca, Anna (August 26, 2019). "Taylor Swift: Lover". Pitchfork. Archived from the original on August 26, 2019. Retrieved April 30, 2020.
Aniftos, Rania (August 16, 2019). "Taylor Swift Unveils 'Lover' Track List". Billboard. Archived from the original on August 17, 2019. Retrieved August 23, 2019.
"RIAS International Top Charts Week 35". Recording Industry Association Singapore. Archived from the original on September 5, 2019.
"Top 20 Most Streamed International & Domestic Singles In Malaysia" (PDF). Recording Industry Association of Malaysia. Retrieved September 12, 2019.[dead link]
"IRMA – Irish Charts". Irish Recorded Music Association. Archived from the original on June 26, 2019. Retrieved September 7, 2019.
"NZ Top 40 Singles Chart". Recorded Music NZ. September 2, 2019. Archived from the original on August 30, 2019. Retrieved September 7, 2019.
"Single Top 50: 8 September 2019". Australian Recording Industry Association. Archived from the original on July 5, 2023. Retrieved July 5, 2023.
"Official Singles Chart Top 100". Official Charts Company. Retrieved September 7, 2019.
"Billboard Canadian Hot 100: Week of September 7, 2019". Billboard. Archived from the original on July 5, 2023. Retrieved July 5, 2023.
Trust, Gary (September 3, 2019). "Every Song From Taylor Swift's 'Lover' Album Charts On The Hot 100". Billboard. Archived from the original on May 8, 2020. Retrieved April 30, 2020.
Zellner, Xander (May 30, 2023). "Taylor Swift's 'Cruel Summer' Returns to Hot 100 for First Time Since 2019". Billboard. Archived from the original on June 1, 2023. Retrieved June 1, 2023.
Lang, Cady (June 16, 2023). "How a 4-Year-Old Track Became Taylor Swift's Song of the Moment". Time. Archived from the original on July 5, 2023. Retrieved June 18, 2023.
Griffiths, George (June 27, 2023). "Inside the Rise and Rise of Taylor Swift's 'Cruel Summer'". Official Charts Company. Archived from the original on July 7, 2023. Retrieved July 10, 2023.
Spanos, Brittany (April 28, 2023). "'Cruel Summer': The Taylor Swift Hit We Deserve". Rolling Stone. Archived from the original on April 29, 2023. Retrieved April 30, 2023.
Trust, Gary (June 15, 2023). "Taylor Swift's 'Cruel Summer' Is Her New Radio Single, Four Years After Its Release". Billboard. Archived from the original on June 15, 2023. Retrieved June 15, 2023.
Tomás Mier; Larisha Paul (June 2, 2023). "Taylor Swift Extends 'Eras Tour' to Latin America, Promises 'Lots More' International Shows". Rolling Stone. Archived from the original on June 24, 2023. Retrieved June 18, 2023.
Pandey, Manish; Allison, Pete (June 30, 2023). "Taylor Swift: Cruel Summer back in UK top 40 again four years later". BBC News. Archived from the original on July 5, 2023. Retrieved July 5, 2023.

    "A Recap of Radio Add Recaps". Hits. June 13, 2023. Archived from the original on March 4, 2024. Retrieved March 3, 2024.
    Cantor, Brian (June 13, 2023). "Taylor Swift's 'Cruel Summer' Earns Most Added Honor At Pop Radio". Headline Planet. Archived from the original on November 30, 2023. Retrieved March 3, 2024.

Cantor, Brian (June 26, 2023). "Taylor Swift's 'Cruel Summer' Ranks As Hot Adult Contemporary Radio's Most Added Song". Headline Planet. Archived from the original on July 8, 2023. Retrieved March 3, 2024.
Iasimone, Ashley (June 18, 2023). "Taylor Swift on 'Cruel Summer' Becoming a Single Four Years After Its Release: 'No One Understands How This Is Happening'". Billboard. Archived from the original on June 18, 2023. Retrieved June 18, 2023.
Scribner, Herb (June 18, 2023). "Taylor Swift has a new single on the way, but it's actually four years old". The Washington Post. ISSN 0190-8286. Archived from the original on June 27, 2023. Retrieved June 22, 2023.
Trust, Gary (July 10, 2023). "Olivia Rodrigo's 'Vampire' Debuts as Her Third Billboard Hot 100 No. 1". Billboard. Archived from the original on July 10, 2023. Retrieved July 10, 2023.
Atkinson, Katie (October 19, 2023). "Taylor Swift Unveils Live Version of 'Cruel Summer' & New Remix 'For Old Time's Sake'". Billboard. Archived from the original on October 19, 2023. Retrieved October 19, 2023.
Trust, Gary (October 23, 2023). "Taylor Swift's 'Cruel Summer' Hits No. 1 on Billboard Hot 100, Becoming Her 10th Leader". Billboard. Archived from the original on October 23, 2023. Retrieved October 23, 2023.
Trust, Gary (November 6, 2023). "Taylor Swift's 'Is It Over Now? (Taylor's Version)' Debuts at No. 1 on Billboard Hot 100". Billboard. Archived from the original on November 6, 2023. Retrieved January 14, 2024.
Trust, Gary (November 13, 2023). "Taylor Swift's 'Cruel Summer' Returns to No. 1 on Hot 100, Jung Kook & The Beatles Debut in Top 10". Billboard. Archived from the original on November 13, 2023. Retrieved November 13, 2023.
Trust, Gary (January 8, 2024). "Jack Harlow's 'Lovin on Me' Returns to No. 1 on Hot 100, Doja Cat's 'Agora Hills' Hits Top 10". Billboard. Archived from the original on January 11, 2024. Retrieved January 14, 2024.
Trust, Gary (October 2, 2023). "Doja Cat's 'Paint the Town Red' Tops Hot 100 for Second Week, Taylor Swift's 'Cruel Summer' Rules Radio". Billboard. Archived from the original on October 2, 2023. Retrieved October 3, 2023.
Trust, Gary (July 28, 2023). "Taylor Swift Breaks Record for Most No. 1s on Pop Airplay Chart As 'Cruel Summer' Becomes Her 12th". Billboard. Archived from the original on September 29, 2023. Retrieved July 28, 2023.
Trust, Gary (August 18, 2023). "'Summer' Love: Taylor Swift Breaks Record for Most No. 1s Among Soloists on Adult Pop Airplay Chart". Billboard. Archived from the original on August 20, 2023. Retrieved August 20, 2023.
Trust, Gary (January 19, 2024). "Taylor Swift's 'Cruel Summer' Hits No. 1 on Adult Contemporary Chart". Billboard. Archived from the original on January 19, 2024. Retrieved January 19, 2024.
Atkinson, Katie; Dailey, Hannah; Denis, Kyle; Lipshutz, Jason; Unterberger, Andrew (July 18, 2023). "Why Did Taylor Swift's Speak Now Re-Recording Become Her Best-Performing Taylor's Version Yet?". Billboard. Archived from the original on July 19, 2023. Retrieved July 19, 2023.
Trust, Gary (March 25, 2024). "Teddy Swims' 'Lose Control' Hits No. 1 on Billboard Hot 100". Billboard. Retrieved March 25, 2024.
Trust, Gary (May 20, 2024). "Taylor Swift's 'Cruel Summer' Is Now Her Sole Longest-Charting Hot 100 Hit". Billboard. Retrieved June 4, 2024.
"Taylor Swift – Cruel Summer". ARIA Top 50 Singles. Retrieved December 17, 2024.
"Taylor Swift Chart History (Canadian Hot 100)". Billboard. Retrieved October 24, 2023.
"RIAS Top Charts Week 24 (9 – 15 Jun 2023)". Recording Industry Association Singapore. Archived from the original on June 20, 2023. Retrieved June 20, 2023.
"NZ Top 40 Singles Chart". Recorded Music NZ. September 11, 2023. Archived from the original on September 14, 2023. Retrieved September 9, 2023.
"Irish Singles Chart: Week Ending 2 June 2023". Irish Recorded Music Association. Archived from the original on February 4, 2021. Retrieved May 31, 2023.
"TOP 20 Most Streamed International Singles In Malaysia Week 35 (18/08/2023- 24/08/2023)". Recording Industry Association of Malaysia. September 2, 2023. Archived from the original on October 9, 2023. Retrieved September 4, 2023 – via Facebook.
Pacilio, Isabela (November 28, 2023). "Hot 100: Taylor Swift pula 40 posições após shows no Brasil com 'Cruel Summer'" [Hot 100: Taylor Swift jumps 40 positions after shows in Brazil with 'Cruel Summer']. Billboard Brasil (in Brazilian Portuguese). Archived from the original on November 28, 2023. Retrieved November 28, 2023.
"Taylor Swift Chart History (Philippines Songs)". Billboard. Archived from the original on November 8, 2023. Retrieved July 11, 2023.
"Taylor Swift Chart History (Indonesia Songs)". Billboard. Retrieved March 20, 2024.
"Danish single certifications – Taylor Swift – Cruel Summer". IFPI Danmark. Retrieved April 30, 2024.
"IFPI Charts – Digital Singles Chart (International)" (in Greek). IFPI Greece. Retrieved October 19, 2023.
"Italian single certifications – Taylor Swift – Cruel Summer" (in Italian). Federazione Industria Musicale Italiana. Retrieved March 18, 2024.
"OLiS - oficjalna lista wyróżnień" (in Polish). Polish Society of the Phonographic Industry. Retrieved July 31, 2024. Click "TYTUŁ" and enter Cruel Summer in the search box.
"Portuguese single certifications – Taylor Swift – Cruel Summer" (PDF) (in Portuguese). Associação Fonográfica Portuguesa. Retrieved October 13, 2024.
"British single certifications – Taylor Swift – Cruel Summer". British Phonographic Industry. Retrieved July 5, 2024.
"Taylor Swift Chart History (Global 200)". Billboard. Retrieved October 31, 2023.
"The Top Songs, Artists, Podcasts, and Listening Trends of 2023 Revealed". Spotify. November 29, 2023. Archived from the original on November 29, 2023. Retrieved November 29, 2023.
Brandle, Lars (February 26, 2023). "Miley Cyrus' 'Flowers' Wins IFPI Global Single Award For 2023". Billboard. Archived from the original on February 26, 2024. Retrieved February 26, 2024.
"2024 Luminate Mid-Year Music Report" (PDF). Luminate Data. Archived (PDF) from the original on July 22, 2024. Retrieved July 25, 2024.
Levine, Nick (April 26, 2024). "St. Vincent: "When I think about music that I love, I don't give a shit what the artist was thinking"". NME. Retrieved April 28, 2024.
Caramanica, Jon (August 23, 2019). "Taylor Swift Emerges From the Darkness Unbroken on 'Lover'". The New York Times. ISSN 0362-4331. Archived from the original on August 24, 2019. Retrieved April 30, 2020.
Wood, Mikael (August 25, 2019). "Taylor Swift's 'Lover': All 18 songs, ranked". Los Angeles Times. Archived from the original on June 8, 2020. Retrieved June 8, 2020.
Abad-Santos, Alex (August 23, 2019). "The 3 best songs on Taylor Swift's new album, Lover". Vox. Archived from the original on July 26, 2020. Retrieved April 30, 2020.
Levine, Nick (August 23, 2019). "Taylor Swift – 'Lover' review". NME. Archived from the original on August 23, 2019. Retrieved April 30, 2020.
Barr, Natalia (August 26, 2019). "Album Review: Taylor Swift Takes the High Road on the More Mature Lover". Consequence. Archived from the original on August 29, 2019. Retrieved May 27, 2020.
"The 50 Best Songs of 2019". Rolling Stone. December 6, 2019. Archived from the original on December 6, 2019. Retrieved December 6, 2019.
"The 100 Best Songs of 2019: Staff List". Billboard. December 11, 2019. Archived from the original on January 31, 2020. Retrieved April 30, 2020.
"The 100 Greatest Song Bridges of the 21st Century: Staff Picks". Billboard. May 13, 2021. Archived from the original on May 18, 2021. Retrieved May 14, 2021.
Sheffield, Rob (October 26, 2021). "All 199 of Taylor Swift's Songs, Ranked by Rob Sheffield". Rolling Stone. Archived from the original on February 15, 2021. Retrieved October 26, 2021.
Mylrea, Hannah (September 8, 2020). "Every Taylor Swift song ranked in order of greatness". NME. Archived from the original on September 17, 2020. Retrieved September 17, 2020.
"Taylor Swift: Her 15 Best Songs". Clash. January 2, 2022. Archived from the original on May 25, 2022. Retrieved January 6, 2022.
Hudson, Alex; LaPierre, Megan (October 20, 2022). "Taylor Swift's 20 Best Songs Ranked". Exclaim!. Archived from the original on December 6, 2022. Retrieved December 6, 2022.
Nuss, Allaire (November 7, 2022). "Taylor Swift's 10 Seminal Albums, Ranked". Entertainment Weekly. Archived from the original on November 26, 2022. Retrieved May 23, 2023.
Unterberger, Andrew (December 15, 2023). "Billboard's Greatest Pop Stars of 2023: No. 1 — Taylor Swift". Billboard. Archived from the original on December 15, 2023. Retrieved December 16, 2023.
"The 500 Greatest Songs of All Time". Rolling Stone. February 16, 2024. Archived from the original on February 16, 2024. Retrieved February 18, 2024.
Atkinson, Katie (April 2, 2024). "Here Are the 2024 iHeartRadio Music Awards Winners". Billboard. Retrieved April 2, 2024.
Grein, Paul (June 5, 2024). "Taylor Swift Wins Songwriter of the Year at 2024 BMI Pop Awards". Billboard. Retrieved June 5, 2024.
Grein, Paul (November 25, 2024). "Zach Bryan, Taylor Swift, Morgan Wallen & Sabrina Carpenter Are Top 2024 Billboard Music Awards Finalists: Full List". Billboard. Retrieved November 26, 2024.
Nesvig, Kara (April 23, 2020). "Taylor Swift Praised "HSM" Series Star Olivia Rodrigo's "Cruel Summer" Cover". Teen Vogue. Archived from the original on May 25, 2022. Retrieved July 16, 2023.
Spanos, Brittany (July 9, 2021). "Olivia Rodrigo Adds Taylor Swift, St. Vincent, Jack Antonoff Co-Writes to 'Deja Vu'". Rolling Stone. Archived from the original on July 9, 2021. Retrieved July 30, 2021.
Longeretta, Emily (June 30, 2022). "The Summer I Turned Pretty Hits No. 1 on Amazon Prime Video, Taylor Swift Songs Re-Enter Top 40 Chart Three Years After Release (Exclusive)". Variety. Archived from the original on July 1, 2022. Retrieved July 2, 2022.
Longeretta, Emily (November 22, 2023). "'Dancing With the Stars' Semi-Finalists Revealed After Taylor Swift Night". Variety. Archived from the original on January 19, 2024. Retrieved January 19, 2024.
"G Flip covers Taylor Swift's 'Cruel Summer' for Like A Version". Triple J. Australian Broadcasting Corporation. January 12, 2024. Archived from the original on January 20, 2024. Retrieved January 21, 2024.
Aniftos, Rania (January 12, 2024). "Teddy Swims Adds the Heat to Taylor Swift's 'Cruel Summer'". Billboard. Archived from the original on January 12, 2024. Retrieved January 19, 2024.
"ČNS IFPI" (in Czech). Hitparáda – Digital Top 100 Oficiální. IFPI Czech Republic. Note: Select 35. týden 2019 in the date selector. Retrieved September 3, 2019.
"Digital Singles Chart (International)". IFPI Greece. September 11, 2019. Archived from the original on September 11, 2019. Retrieved June 6, 2022.
"Official Irish Singles Chart Top 50". Official Charts Company. Retrieved June 27, 2023.
"Single Top 40: 2 September 2019". Recorded Music NZ. Archived from the original on September 28, 2023. Retrieved July 5, 2023.
"Taylor Swift – Cruel Summer". AFP Top 100 Singles. Retrieved September 4, 2019.
"Official Scottish Singles Sales Chart Top 100". Official Charts Company. Retrieved September 7, 2019.
"ČNS IFPI" (in Slovak). Hitparáda – Singles Digital Top 100 Oficiálna. IFPI Czech Republic. Note: Select 35. týden 2019 in the date selector. Retrieved September 3, 2019.
"Veckolista Heatseeker, vecka 35" (in Swedish). Sverigetopplistan. Archived from the original on March 22, 2020. Retrieved September 7, 2019.
"Billboard Hot 100: Week of September 7, 2019". Billboard. Archived from the original on November 3, 2022. Retrieved June 27, 2023.
"Taylor Swift – Chart History (Argentina Hot 100)" Billboard Argentina Hot 100 Singles for Taylor Swift. Retrieved January 14, 2024.
"Taylor Swift – Cruel Summer" (in German). Ö3 Austria Top 40. Retrieved November 1, 2023.
"Taylor Swift – Cruel Summer" (in Dutch). Ultratop 50. Retrieved October 29, 2023.
"Taylor Swift Chart History (Brasil Hot 100)". Billboard. Retrieved November 30, 2023.
"Taylor Swift Chart History (Canada AC)". Billboard. Retrieved November 4, 2023.
"Taylor Swift Chart History (Canada CHR/Top 40)". Billboard. Retrieved August 17, 2023.
"Taylor Swift Chart History (Canada Hot AC)". Billboard. Retrieved February 10, 2024.
Taylor Swift — Cruel Summer. TopHit. Retrieved October 6, 2023.
"ARC 100 - date: 18. September 2023" (PDF). Hrvatska radiotelevizija. Archived (PDF) from the original on November 26, 2023. Retrieved September 23, 2023.
"ČNS IFPI" (in Czech). Hitparáda – Radio Top 100 Oficiální. IFPI Czech Republic. Note: Select 43. týden 2023 in the date selector. Retrieved October 30, 2023.
"ČNS IFPI" (in Czech). Hitparáda – Digital Top 100 Oficiální. IFPI Czech Republic. Note: Select 34. týden 2023 in the date selector. Retrieved August 28, 2023.
"Taylor Swift – Cruel Summer". Tracklisten. Retrieved November 5, 2023.
"Taylor Swift: Cruel Summer" (in Finnish). Musiikkituottajat. Retrieved October 24, 2023.
"Top Singles (Week 11, 2024)" (in French). Syndicat National de l'Édition Phonographique. Retrieved March 18, 2024.
"Classement Radio" (in French). Syndicat National de l'Édition Phonographique. Archived from the original on August 9, 2022. Retrieved February 5, 2024.
"Taylor Swift – Cruel Summer" (in German). GfK Entertainment charts. Retrieved October 27, 2023.
"Airplay Charts Deutschland". Hung Medien. Retrieved June 10, 2024.
"Digital Singles Chart (International) – Week: 43/2023". IFPI Greece. Archived from the original on October 26, 2023. Retrieved November 1, 2023.
"Taylor Swift Chart History (Hong Kong Songs)". Billboard. Retrieved March 12, 2024.
"Tónlistinn – Lög" [The Music – Songs] (in Icelandic). Plötutíðindi. Archived from the original on October 28, 2023. Retrieved October 28, 2023.
"IMI International Top 20 Singles for week ending 13th November 2023 | Week 45 of 52". IMIcharts. Archived from the original on November 16, 2023.
"Official Irish Singles Chart Top 50". Official Charts Company. Retrieved November 5, 2023.
"Media Forest charts". Media Forest. Archived from the original on August 7, 2023. Retrieved August 7, 2023.
"Taylor Swift – Cruel Summer". Top Digital Download. Retrieved November 5, 2023.
"Taylor Swift Chart History (Japan Hot 100)". Billboard. Retrieved February 21, 2024.
"Oricon Top 50 Combined Singles: 2024-02-26" (in Japanese). Oricon. Archived from the original on February 22, 2024. Retrieved February 22, 2024.
"EHR TOP 40 - 2023.09.08". European Hit Radio. Archived from the original on September 9, 2023. Retrieved September 9, 2023.
"Mūzikas Patēriņa Tops/ 43. nedēļa". LAIPA. October 31, 2023. Archived from the original on October 31, 2023. Retrieved October 31, 2023.
"Latvijas radio stacijās spēlētākās dziesmas TOP 36. nedēļa" (in Latvian). Latvian Music Producers Association. September 11, 2023. Archived from the original on October 10, 2023. Retrieved September 12, 2023.
"2023 43-os savaitės klausomiausi (Top 100)" (in Lithuanian). AGATA. October 27, 2023. Archived from the original on November 1, 2023. Retrieved October 27, 2023.
"Taylor Swift Chart History (Luxembourg Songs)". Billboard. Archived from the original on November 1, 2022. Retrieved October 27, 2023.
"Taylor Swift Chart History (Malaysia Songs)". Billboard. Retrieved March 12, 2024.
"TOP 20 Most Streamed International Singles In Malaysia Week 10 (01/03/2024-07/03/2024)". RIM. March 16, 2024. Retrieved March 16, 2024 – via Facebook.
"This Week's Official MENA Chart Top 20: from 20/10/2023 to 26/10/2023". International Federation of the Phonographic Industry. October 20, 2023. Archived from the original on November 1, 2023. Retrieved November 1, 2023.
"Nederlandse Top 40 – week 44, 2023" (in Dutch). Dutch Top 40. Retrieved November 4, 2023.
"Nederlandse Global Top 40 – week 29, 2023" (in Dutch). Dutch Top 40. Archived from the original on October 8, 2023. Retrieved September 2, 2023.
"Taylor Swift – Cruel Summer" (in Dutch). Single Top 100. Retrieved November 11, 2023.
"Taylor Swift – Cruel Summer". Top 40 Singles. Retrieved November 5, 2023.
"TurnTable Nigeria Top 100: December 1st, 2023 - December 7th, 2023". TurnTable. Archived from the original on December 14, 2023. Retrieved December 14, 2023.
"Taylor Swift – Cruel Summer". VG-lista. Retrieved November 5, 2023.
"Top 20 Panamá General - Del 28 de Agosto al 3 de Septiembre, 2023". Monitor Latino. Archived from the original on September 15, 2023. Retrieved September 10, 2023.
"TOP 50 INTERNACIONAL BMAT-PRODUCE DEL 5 A EL 11 DE OCTUBRE 2023" (in Spanish). Sociedad Panameña de Productores Fonográficos. Archived from the original on October 11, 2023. Retrieved October 12, 2023.
"Top 20 Paraguay General - Del 11 al 17 de Septiembre, 2023". Monitor Latino. Archived from the original on October 2, 2023. Retrieved September 19, 2023.
"OLiS – oficjalna lista airplay" (Select week 14.10.2023–20.10.2023.) (in Polish). OLiS. Archived from the original on October 23, 2023. Retrieved October 23, 2023.
"OLiS – oficjalna lista sprzedaży – single w streamie" (Select week 20.10.2023–26.10.2023.) (in Polish). OLiS. Archived from the original on February 2, 2023. Retrieved November 2, 2023.
"Taylor Swift – Cruel Summer". AFP Top 100 Singles. Retrieved November 3, 2023.
"RIAS Top Charts Week 27 (30 Jun - 6 Jul 2023)". RIAS. Archived from the original on July 11, 2023. Retrieved July 11, 2023.
"ČNS IFPI" (in Slovak). Hitparáda – Radio Top 100 Oficiálna. IFPI Czech Republic. Note: Select 40. týden 2023 in the date selector. Retrieved October 9, 2023.
"ČNS IFPI" (in Slovak). Hitparáda – Singles Digital Top 100 Oficiálna. IFPI Czech Republic. Note: Select 39. týden 2023 in the date selector. Retrieved October 2, 2023.
"Local & International Radio Chart Top 10 Week 50 – 2023". The Official South African Charts. Archived from the original on December 22, 2023. Retrieved December 22, 2023.
"Digital Chart – Week 33 of 2024". Circle Chart (in Korean). Retrieved August 22, 2024.
"Top 100 Canciones". PROMUSICAE. Retrieved June 16, 2024.
"Veckolista Singlar, vecka 21". Sverigetopplistan. Retrieved May 24, 2024.
"Taylor Swift – Cruel Summer". Swiss Singles Chart. Retrieved July 14, 2024.
"Taylor Swift Chart History (Taiwan Songs)". Billboard. Archived from the original on November 1, 2022. Retrieved June 3, 2024.
"This Week's Official UAE Chart Top 20: from 20/10/2023 to 26/10/2023". International Federation of the Phonographic Industry. October 20, 2023. Archived from the original on November 1, 2023. Retrieved November 1, 2023.
"Official Singles Chart Top 100". Official Charts Company. Retrieved September 8, 2023.
"Taylor Swift Chart History (Hot 100)". Billboard. Retrieved October 23, 2023.
"Taylor Swift Chart History (Adult Contemporary)". Billboard. Retrieved January 20, 2023.
"Taylor Swift Chart History (Adult Pop Songs)". Billboard. Retrieved August 21, 2023.
"Taylor Swift Chart History (Dance Mix/Show Airplay)". Billboard. Retrieved November 19, 2023.
"Taylor Swift Chart History (Pop Songs)". Billboard. Retrieved July 28, 2023.
"Top 100 - Record Report". Record Report. Archived from the original on December 21, 2023. Retrieved December 21, 2023.
"Taylor Swift Chart History (Billboard Vietnam Hot 100)". Billboard. Archived from the original on June 6, 2022. Retrieved July 15, 2023.
"Top Radio Hits Global Monthly Chart October 2023". TopHit. Archived from the original on November 4, 2023. Retrieved November 5, 2023.
"Diciembre - TOP #100 de Canciones de SGP" (in Spanish). Sociedad de Gestion de Productores Fonograficos del Paraguay. Archived from the original on January 6, 2024. Retrieved January 6, 2024.
"Digital Chart – August 2024". Circle Chart (in Korean). Retrieved September 12, 2024.
"ARIA Top 100 Singles Chart for 2023". Australian Recording Industry Association. Archived from the original on January 12, 2024. Retrieved January 12, 2024.
"Ö3 Austria Top40 Jahrescharts 2023: Singles". Ö3 Austria Top 40. November 8, 2019. Archived from the original on December 28, 2023. Retrieved December 28, 2023.
"Canadian Hot 100 – Year-End 2023". Billboard. Archived from the original on November 21, 2023. Retrieved November 22, 2023.
"Jahrescharts 2023" (in German). GfK Entertainment charts. Archived from the original on December 10, 2023. Retrieved December 10, 2023.
"Billboard Global 200 – Year-End 2023". Billboard. Archived from the original on November 21, 2023. Retrieved November 22, 2023.
"Miley Cyrus' Flowers Confirmed by IFPI as Biggest-Selling Global Single of the Year". IFPI. February 26, 2024. Archived from the original on February 26, 2024. Retrieved February 26, 2024.
"TÓNLISTINN – LÖG – 2023" (in Icelandic). Plötutíðindi. Retrieved March 8, 2024.
"Top 100-Jaaroverzicht van 2023" (in Dutch). Dutch Top 40. Archived from the original on December 31, 2023. Retrieved January 1, 2024.
"Jaaroverzichten – Single 2023". dutchcharts.nl (in Dutch). Archived from the original on January 4, 2024. Retrieved January 3, 2024.
"Top Selling Singles of 2023". Recorded Music NZ. Archived from the original on December 20, 2023. Retrieved December 22, 2023.
"Billboard Philippines Songs – Year-End 2023". Billboard. Archived from the original on December 16, 2023. Retrieved November 25, 2023.
"Podsumowanie roczne – OLiA" (in Polish). OLiS. Polish Society of the Phonographic Industry. Retrieved March 28, 2024.
"Årslista Singlar, 2023". Sverigetopplistan. Archived from the original on January 18, 2024. Retrieved January 18, 2024.
"Schweizer Jahreshitparade 2023". hitparade.ch. Archived from the original on January 2, 2024. Retrieved December 31, 2023.
Griffiths, George (December 29, 2023). "The Official Top 40 Biggest Songs of 2023". Official Charts Company. Archived from the original on December 29, 2023. Retrieved December 29, 2023.
"Hot 100 Songs – Year-End 2023". Billboard. Archived from the original on November 21, 2023. Retrieved November 21, 2023.
"Adult Contemporary Songs — Year-End 2023". Billboard. Archived from the original on November 22, 2023. Retrieved November 21, 2023.
"Adult Pop Airplay Songs — Year-End 2023". Billboard. Archived from the original on November 10, 2022. Retrieved November 21, 2023.
"Pop Airplay Songs — Year-End 2023". Billboard. Archived from the original on November 21, 2023. Retrieved November 21, 2023.
"Jahreshitparade Singles 2024". austriancharts.at (in German). Retrieved January 3, 2025.
"Canadian Hot 100 – Year-End 2024". Billboard. Archived from the original on December 13, 2024. Retrieved December 13, 2024.
"Top Radio Hits Estonia Annual Chart: 2024". TopHit. Retrieved January 3, 2025.
"Top de l'année – Top Singles – 2024" (in French). Syndicat National de l'Édition Phonographique. Retrieved January 11, 2025.
"Top 100 Single-Jahrescharts" (in German). GfK Entertainment charts. Retrieved December 9, 2024.
"Billboard Global 200 – Year-End 2024". Billboard. Retrieved December 14, 2024.
"TÓNLISTINN – LÖG – 2024" (in Icelandic). Plötutíðindi. Retrieved January 11, 2025.
"Top 100-Jaaroverzicht van 2024" (in Dutch). Dutch Top 40. Retrieved December 30, 2024.
"Jaaroverzichten – Single 2024". dutchcharts.nl (in Dutch). Retrieved January 3, 2025.
"End of Year Top 50 Singles". Recorded Music NZ. Archived from the original on December 20, 2024. Retrieved December 21, 2024.
"Philippines Hot 100 – Year of 2024". Billboard Philippines. Retrieved December 21, 2024.
"Digital Chart – 2024". Circle Chart (in Korean). Retrieved January 11, 2025.
"Schweizer Jahreshitparade 2024". hitparade.ch (in German). Retrieved December 30, 2024.
Ainsley, Helen (December 27, 2024). "The Official biggest Songs of 2024". Official Charts Company. Retrieved December 27, 2024.
"Hot 100 Songs – Year-End 2024". Billboard. Archived from the original on December 13, 2024. Retrieved December 13, 2024.
"Adult Contemporary Songs — Year-End 2024". Billboard. Archived from the original on December 13, 2024. Retrieved December 13, 2024.
"Adult Pop Airplay Songs — Year-End 2024". Billboard. Archived from the original on December 13, 2024. Retrieved December 13, 2024.
"Pop Airplay Songs — Year-End 2024". Billboard. Archived from the original on December 13, 2024. Retrieved December 13, 2024.
"Resumen Anual 2024" (PDF). Record Report. Retrieved December 26, 2024.
"ARIA Charts – Accreditations – 2024 Singles" (PDF). Australian Recording Industry Association. Retrieved April 28, 2024.
"Austrian single certifications – Taylor Swift – Cruel Summer" (in German). IFPI Austria. Retrieved May 29, 2024.
"Brazilian single certifications – Taylor Swift – Cruel Summer" (in Portuguese). Pro-Música Brasil. Retrieved July 22, 2024.
"French single certifications – Taylor Swift – Cruel Summer" (in French). Syndicat National de l'Édition Phonographique. Retrieved November 18, 2024.
"Gold-/Platin-Datenbank (Taylor Swift; 'Cruel Summer')" (in German). Bundesverband Musikindustrie. Retrieved March 15, 2024.
"New Zealand single certifications – Taylor Swift – Cruel Summer". Recorded Music NZ. Retrieved November 20, 2024.
"Spanish single certifications – Taylor Swift – Cruel Summer". El portal de Música. Productores de Música de España. Retrieved October 13, 2024.
"The Official Swiss Charts and Music Community: Awards ('Cruel Summer')". IFPI Switzerland. Hung Medien. Retrieved November 29, 2023.
"Japanese single streaming certifications – Taylor Swift – Cruel Summer" (in Japanese). Recording Industry Association of Japan. Retrieved April 23, 2024. Select 2024年3月 on the drop-down menu
"Sverigetopplistan – Taylor Swift" (in Swedish). Sverigetopplistan. Retrieved June 19, 2024.
"South Korean single streaming certifications – Taylor Swift – Cruel Summer" (in Korean). Korea Music Content Association (KMCA). Retrieved December 24, 2024.
"Miley Cyrus' Flowers Confirmed by IFPI as Biggest-Selling Global Single of the Year". IFPI. February 26, 2024. Retrieved April 14, 2024.

    "EarOne | Radio Date, le novità musicali della settimana". EarOne (in Italian). Archived from the original on October 27, 2023. Retrieved October 28, 2023.

    vte

Taylor Swift songs

    Singles discography

Taylor Swift	

    "Tim McGraw" "Picture to Burn" "Teardrops on My Guitar" "A Place in This World" "Should've Said No" "Our Song"

Fearless	

    "Fearless" "Fifteen" "Love Story" "Hey Stephen" "White Horse" "You Belong with Me" "Breathe" "You're Not Sorry" "The Way I Loved You" "Forever & Always" "The Best Day" "Change"

Taylor's Version	

    "You All Over Me" "Mr. Perfectly Fine" "That's When"

Speak Now	

    "Mine" "Sparks Fly" "Back to December" "Speak Now" "Dear John" "Mean" "The Story of Us" "Never Grow Up" "Enchanted" "Better than Revenge" "Innocent" "Haunted" "Long Live" "Ours"

Taylor's Version	

    "Electric Touch" "When Emma Falls in Love" "I Can See You" "Castles Crumbling"

Red	

    "State of Grace" "Red" "Treacherous" "I Knew You Were Trouble" "All Too Well" "22" "We Are Never Ever Getting Back Together" "The Last Time" "Holy Ground" "Everything Has Changed" "Begin Again"

Taylor's Version	

    "Nothing New" "Message in a Bottle" "I Bet You Think About Me" "Forever Winter"

1989	

    "Welcome to New York" "Blank Space" "Style" "Out of the Woods" "All You Had to Do Was Stay" "Shake It Off" "I Wish You Would" "Bad Blood" "Wildest Dreams" "How You Get the Girl" "This Love" "I Know Places" "Clean" "You Are in Love" "New Romantics"

Taylor's Version	

    "'Slut!'" "Say Don't Go" "Now That We Don't Talk" "Suburban Legends" "Is It Over Now?"

Reputation	

    "...Ready for It?" "End Game" "I Did Something Bad" "Don't Blame Me" "Delicate" "Look What You Made Me Do" "So It Goes..." "Gorgeous" "Getaway Car" "Dress" "Call It What You Want" "New Year's Day"

Lover	

    "I Forgot That You Existed" "Cruel Summer" "Lover" "The Man" "The Archer" "Miss Americana & the Heartbreak Prince" "Paper Rings" "Cornelia Street" "Death by a Thousand Cuts" "London Boy" "Soon You'll Get Better" "False God" "You Need to Calm Down" "Me!"

Folklore	

    "The 1" "Cardigan" "The Last Great American Dynasty" "Exile" "My Tears Ricochet" "Mirrorball" "Seven" "August" "This Is Me Trying" "Illicit Affairs" "Invisible String" "Mad Woman" "Epiphany" "Betty" "Peace" "Hoax" "The Lakes"

Evermore	

    "Willow" "Champagne Problems" "'Tis the Damn Season" "Tolerate It" "No Body, No Crime" "Happiness" "Dorothea" "Coney Island" "Long Story Short" "Marjorie"

Midnights	

    "Lavender Haze" "Maroon" "Anti-Hero" "Snow on the Beach" "You're on Your Own, Kid" "Midnight Rain" "Question...?" "Vigilante Shit" "Bejeweled" "Labyrinth" "Karma" "Sweet Nothing" "Mastermind" "Hits Different" "Bigger Than the Whole Sky" "Would've, Could've, Should've" "You're Losing Me"

The Tortured Poets
Department	

    "Fortnight" "The Tortured Poets Department "My Boy Only Breaks His Favorite Toys" "Down Bad" "So Long, London" "But Daddy I Love Him" "Fresh Out the Slammer" "Florida!!!" "Guilty as Sin?" "Who's Afraid of Little Old Me?" "Loml" "I Can Do It with a Broken Heart" "The Smallest Man Who Ever Lived" "The Alchemy" "Clara Bow" "The Black Dog" "So High School" "Thank You Aimee"

Soundtrack songs	

    "Crazier" "Today Was a Fairytale" "Safe & Sound" "Eyes Open" "Sweeter than Fiction" "I Don't Wanna Live Forever" "Beautiful Ghosts" "Only the Young" "Carolina"

Featured songs	

    "Two Is Better Than One" "Half of My Heart" "Both of Us" "Babe" "Gasoline" "Renegade" "The Joker and the Queen" "The Alcott" "Us"

Other songs	

    "Best Days of Your Life" "Ronan" "Highway Don't Care" "Better Man" "Christmas Tree Farm" "All of the Girls You Loved Before"

    Category


Authority control databases Edit this at Wikidata	

    MusicBrainz workMusicBrainz release group

Categories:

    2019 songs2023 singlesAmerican synth-pop songsBillboard Global 200 number-one singlesBillboard Hot 100 number-one singlesCanadian Hot 100 number-one singlesIndustrial songsElectropop songsNumber-one singles in AustraliaNumber-one singles in LatviaNumber-one singles in the PhilippinesNumber-one singles in SingaporeRepublic Records singlesSongs about fameSongs written by Taylor SwiftSongs written by Jack AntonoffSongs written by St. Vincent (musician)Song recordings produced by Taylor SwiftSong recordings produced by Jack AntonoffTaylor Swift songs

    This page was last edited on 11 January 2025, at 10:01 (UTC).
    Text is available under the Creative Commons Attribution-ShareAlike 4.0 License; additional terms may apply. By using this site, you agree to the Terms of Use and Privacy Policy. Wikipedia® is a registered trademark of the Wikimedia Foundation, Inc., a non-profit organization.

    Privacy policy
    About Wikipedia
    Disclaimers
    Contact Wikipedia
    Code of Conduct
    Developers
    Statistics
    Cookie statement
    Mobile view

    Wikimedia Foundation
    Powered by MediaWiki



Wikipedia The Free Encyclopedia

    Donate
    Create account
    Log in

Contents
(Top)
Background

Music and lyrics

Release

Critical reception

Commercial performance

Music video

Accolades

Live performances

Controversies

Cover versions and usage in media

Personnel

Charts

Certifications

Release history

"Shake It Off (Taylor's Version)"

    See also
    Footnotes
    References
    Bibliography

Shake It Off

    Article
    Talk

    Read
    Edit
    View history

Tools

Appearance
Text

    Small
    Standard
    Large

Width

    Standard
    Wide

Color (beta)

    Automatic
    Light
    Dark

Featured article
From Wikipedia, the free encyclopedia
For other uses, see Shake It Off (disambiguation).
"Shake It Off"
Cover artwork of "Shake It Off", a polaroid photo of Swift dancing
Single by Taylor Swift
from the album 1989
Released	August 19, 2014
Studio	

    MXM (Stockholm, Sweden)
    Conway Recording (Los Angeles)

Genre	Dance-pop
Length	3:39
Label	Big Machine
Songwriter(s)	

    Taylor Swift Max Martin Shellback

Producer(s)	

    Max Martin Shellback

Taylor Swift singles chronology
"The Last Time"
(2013) 	"Shake It Off"
(2014) 	"Blank Space"
(2014)
Music video
"Shake It Off" on YouTube

"Shake It Off" is a song by the American singer-songwriter Taylor Swift and the lead single from her fifth studio album, 1989. She wrote the song with its producers, Max Martin and Shellback. Inspired by the media scrutiny on Swift's public image, the lyrics are about her indifference to detractors and their negative remarks. An uptempo dance-pop song, it features a looping drum beat, a saxophone line, and a handclap–based bridge. Big Machine Records released "Shake It Off" on August 19, 2014, to market 1989 as Swift's first pop album after her previous country–styled sound.

Initial reviews mostly praised the catchy production, but some criticized the lyrics as weak and shallow. Retrospectively, critics have considered "Shake It Off" an effective opener for 1989 as an album that transformed Swift's image from country to pop; it was ranked among the best songs of the 2010s decade by NME and Consequence. The single topped charts and was certified multi-platinum in Australia, Canada, and New Zealand, and it was certified platinum in countries across Europe, the Americas, and Asia–Pacific. In the United States, the single peaked atop the Billboard Hot 100 and received a Diamond certification from the Recording Industry Association of America.

Mark Romanek directed the music video for "Shake It Off", which portrays Swift as a clumsy person unsuccessfully attempting several dance moves. Critics accused the video of cultural appropriation for featuring dances associated with people of color such as twerking. Swift performed the song on three of her world tours: the 1989 World Tour (2015), the Reputation Stadium Tour (2018), and the Eras Tour (2023–2024). "Shake It Off" won Favorite Song at the 2015 People's Choice Awards and received three nominations at the 2015 Grammy Awards. Following the 2019 dispute regarding the ownership of Swift's back catalog, she re-recorded the song as "Shake It Off (Taylor's Version)" for her 2023 re-recorded album 1989 (Taylor's Version).
Background

Taylor Swift had been known as a primarily country singer-songwriter until her fourth studio album Red (released in October 2012),[1] which incorporates various pop and rock styles, transcending the country sound of her previous releases.[2] The collaborations with Swedish pop producers Max Martin and Shellback introduced straightforward pop hooks and new genres, including electronic and dubstep, to Swift's discography.[3][4] Swift and her label, Big Machine, promoted it as a country album.[5] The album's diverse musical styles sparked a media debate over her status as a country artist, to which she replied in an interview with The Wall Street Journal, "I leave the genre labeling to other people."[6] Swift began recording her fifth studio album, 1989, while touring to support Red in mid-2013.[7] Inspired by 1980s synth-pop, she conceived 1989 as her first "official pop" record that would transform her image from country to pop.[8][9] Martin and Shellback produced seven out of thirteen tracks for the album's standard edition, including "Shake It Off".[10]
Music and lyrics

Swift wrote the lyrics to "Shake It Off" and composed the song's melody with Martin and Shellback.[11] The last song recorded for 1989,[12] it was recorded by Sam Holland at Conway Recording Studios in Los Angeles and by Michael Ilbert at MXM Studios in Stockholm, Sweden.[10] The track was mixed by Serban Ghenea at MixStar Studios in Virginia Beach, Virginia, and mastered by Tom Coyne at Sterling Sound Studio in New York City.[10]
"Shake It Off"
Duration: 20 seconds.0:20
Featuring a saxophone line, "Shake It Off" is an uptempo dance-pop song that sees Swift expressing disinterest in her detractors' negative remarks. The lyrics were inspired by Swift's experience with the media scrutiny.
Problems playing this file? See media help.

Musically, "Shake It Off" is an uptempo dance-pop song that incorporates a saxophone line.[13][14] Jonas Thander, the song's saxophone player, based his part on Martin's pre-recorded MIDI horn sample, using a tenor horn.[15] It took Thander over ten hours to edit the saxophone part, which he completed over the following day.[15] "Shake It Off" is in G Mixolydian – the fifth mode of the C major scale. It follows a ii–IV–I chord progression (Am–C–G); it employs a verse–prechorus–chorus form to begin with a loose verse, tighten for the prechorus, and loosen again for the chorus.[16] The song's upbeat production is accompanied by a looping drum beat, a handclap–based bridge, and synthesized saxophones.[17][18]

The lyrics of the song were inspired by the media scrutiny that Swift had experienced during her rise to stardom.[19] In an interview with Rolling Stone in August 2014, Swift said about the song's inspiration: "I've had every part of my life dissected ... When you live your life under that kind of scrutiny, you can either let it break you, or you can get really good at dodging punches. And when one lands, you know how to deal with it. And I guess the way that I deal with it is to shake it off."[20] Discussing the song's message with NPR in October 2014, Swift said that "Shake It Off" represented her more mature perspectives from her previous single "Mean" (2010), which was also inspired by her detractors.[21] According to Swift, if "Mean" was where she assumed victimhood, "Shake It Off" found her in a proactive stance to "take back the narrative, and have ... a sense of humor about people who kind of get under [her] skin – and not let them get under [her] skin".[21]

In the first verse of the song, Swift references her perceived image as a flirtatious woman with numerous romantic attachments: "I go on too many dates / But I can't make 'em stay / At least that's what people say."[22][23] The lines in the chorus are arranged rhythmically to produce a catchy hook: "Cause the players gonna play, play, play, play, play / And the haters gonna hate, hate, hate, hate, hate / Baby, I'm just gonna shake, shake, shake, shake, shake."[23][24] The spoken-word bridge opens with Swift asserting that the "dirty cheats of the world ... could have been getting down to this sick beat".[25][26] The lyric "this sick beat" is trademarked to Swift by the U.S. Patent and Trademark Office.[27]
Release

On August 13, 2014, Swift appeared on The Tonight Show Starring Jimmy Fallon, where she announced she would hold a live stream via Yahoo! on August 18, 2014.[28][29] During the live stream, Swift announced the details of the album 1989. She debuted "Shake It Off" as the album's lead single and premiered the song's music video simultaneously.[30] "Shake It Off" was released digitally worldwide by Big Machine on August 19.[31] The same day, Big Machine, in partnership with Republic Records, released the song to US radio.[32] A limited CD single edition was available on September 11.[33] In Europe, "Shake It Off" was added to a BBC Radio playlist on August 25,[34] Italian radio on August 29,[35] and was released as a CD single in Germany on October 10.[36]

The release of "Shake It Off" and its parent album 1989 had been highly anticipated, given Swift's announcement that she would abandon her country roots to release an "official pop" album.[23] The magazine Drowned in Sound described the single as "undoubtedly ... the most significant cultural event" since Radiohead's 2011 album The King of Limbs.[17] While noting that "Shake It Off" was not Swift's first "straight-up pop" song, Billboard's Jason Lipshutz considered it a sign of a "bold foray into the unknown", in which Swift could experiment beyond her well-known formulaic country pop songs that had been critically and commercially successful.[9]
Critical reception

"Shake It Off" received mixed reviews upon release.[37] Although positive reviews found the production catchy, critics deemed the track repetitive and lacking substance compared to Swift's works on Red.[26][38] Randall Roberts from the Los Angeles Times's lauded the sound as "perfect pop confection" but found the lyrics shallow, calling them insensible to the political events at the time: "When lives are at stake and nothing seems more relevant than getting to the Actual Truth, liars and cheats can't and shouldn't be shaken off."[25] In congruence, The Guardian's Molly Fitzpatrick wrote that the lyrics fell short of Swift's songwriting abilities.[39]

Giving the song a three-out-of-five-stars score, Jeff Terich from American Songwriter regarded Swift's new direction as "a left-turn worth following". While Terich agreed that the lyrics were dismissive, he felt that critics should not have taken the song seriously because it was "pretty harmless".[18] In a positive review, Jason Lipshutz from Billboard wrote: "Swift proves why she belongs among pop's queen bees ... the song sounds like a surefire hit."[40] In a review of the album 1989, Alexis Petridis praised the lyrics for "twisting clichés until they sound original".[41] In the words of Andrew Unterberger from Spin, while "Shake It Off" was musically a "red herring" that feels out of place on the album, it thematically represents Swift's new attitude on 1989, where she liberated herself from overtly romantic struggles to embrace positivity.[42] Swift herself acknowledged the song as an outlier on 1989, and deliberately released it as the lead single to encourage audiences to explore the entire album and not just the singles.[43]

Retrospectively, Hannah Mylrea from NME considered "Shake It Off" an effective opener for Swift's 1989 era, which transformed her image to mainstream pop.[44] While saying that "Shake It Off" was not one of the album's better songs, Rob Sheffield from Rolling Stone applauded it for "serving as a trailer to announce her daring Eighties synth-pop makeover".[45] Nate Jones from Vulture agreed, but described the song's bridge as "the worst 24 seconds of the entire album".[26] In his 2019 ranking of Swift's singles, Petridis ranked "Shake It Off" third—behind "Blank Space" (2014) and "Love Story" (2008), lauding its "irresistible" hook and "sharp-tongued wit".[46] Jane Song from Paste was less enthusiastic, placing "Shake It Off" among Swift's worst songs in her catalog: "Swift has a pattern of choosing the worst song from each album as the lead single."[47]
Commercial performance

"Shake It Off" gained an audience of nine million on US airplay after one day of release to radio[48] and debuted at number 45 on Radio Songs after two days of release.[49] After its first week of release, the single debuted at number nine on Adult Top 40 and number 12 on Pop Songs, setting the record for the highest debut on both charts.[50] On the Pop Songs chart, it tied with Mariah Carey's "Dreamlover" (1993) for the highest first-week chart entry.[50] Although not officially released to country radio, the single debuted and peaked at number 58 on Country Airplay.[51]

"Shake It Off" debuted at number one on the US Billboard Hot 100 chart dated September 6, 2014, the 22nd song to do so.[52] After two consecutive weeks at number one, it dropped to number two, where it stayed for eight consecutive weeks.[53] "Shake It Off" returned to number one in its tenth charting week, and spent a further week at number one, totaling four non-consecutive weeks atop the Hot 100.[54] It also topped Billboard airplay-focused charts including Pop Songs, Adult Top 40, and Adult Contemporary.[55] "Shake It Off" was one of the best-selling singles of the 2010s decade in the United States, selling 5.4 million digital copies as of January 2020.[56] As of February 2024, the single remains Swift's biggest hit on the Hot 100, where it spent nearly six months in the 10 ten and 50 weeks in the top 100.[57][58] The song was certified Diamond by the Recording Industry Association of America, which denotes 10 million units based on sales and streams.[59] With this achievement, Swift is the first female artist to have both a song and an album (Fearless) certified Diamond in the United States.[60]

"Shake It Off" also topped the charts and received multi-platinum certifications in Australia (eighteen-times platinum),[61] Canada (six-times platinum),[62] and New Zealand (five-times platinum).[63] In the United Kingdom, it peaked at number two on the UK Singles Chart and, by November 2022, became the first song since 2020 to surpass one million in pure sales.[64] It was certified five-times platinum to become Swift's best-selling single in the United Kingdom as of April 2024.[65][66] In Japan, "Shake It Off" peaked at number four on the Japan Hot 100 and was certified triple platinum.[67] The single also topped record charts in Hungary and Poland,[68] and it was a top-five hit in other European countries, peaking at number two in Spain;[69] number three in Ireland,[70] Norway[71] and Sweden;[72] number four in Denmark[73] and Israel;[74] and number five in Germany[75] and the Netherlands.[76] It was certified triple diamond in Brazil,[77] and double platinum in Austria,[78] Italy,[79] Norway,[80] and Spain.[81]
Music video
Concept
Taylor Swift in the "Shake It Off" music video wearing a black turtleneck
Poster of the 1957 film Funny Face, featuring Audrey Hepburn in a black turtleneck
Swift's black turtleneck and jeans in "Shake It Off" (left) drew comparisons to Audrey Hepburn's outfit in the 1957 film Funny Face (right).[82][83]

The music video for "Shake It Off", directed by Mark Romanek, was released on August 18, 2014, the same day as the song's release.[84] It was shot over three days in June 2014 in Los Angeles.[20] Swift conceived the video as a humorous depiction of her trying to find her identity: "It takes a long time to figure out who you are and where you fit in in the world."[20] To this end, the video depicts Swift as a clumsy person who unsuccessfully attempts dance moves with professional artists, including ballerinas, street dancers, cheerleaders, rhythmic gymnasts and performance artists.[20][84] She summed up the video: "I'm putting myself in all these awkward situations where the dancers are incredible, and I'm having fun with it, but not fitting in ... I'm being embarrassingly bad at it. It shows you to keep doing you, keep being you, keep trying to figure out where you fit in in the world, and eventually you will."[20]

The dances were choreographed by Tyce Diorio.[85] The video's final scenes feature Swift dancing with her fans, who had been handpicked by Swift through social media engagement.[86] The video contains references to other areas of popular culture. According to VH1, those references are: the ballerinas to the 2010 film Black Swan, the breakdancers to the 2010 film Step Up 3D, the "sparkling suits and robotic dance moves" to the electronic music duo Daft Punk, the twerking dance moves to the singer Miley Cyrus, the cheerleaders to Toni Basil's 1981 video "Mickey", and Swift's black turtleneck and jeans to the outfits of Audrey Hepburn in the 1957 film Funny Face.[82] Publications including the Los Angeles Times and The Sydney Morning Herald also noted references to Lady Gaga and Skrillex.[14][24]
Analysis and reception

Molly Fitzpatrick of The Guardian considered Swift "a little too skilled a dancer" for the video's concept, writing: "The incongruent blend of modern dance, ballet, and breakdancing is fun, but the conceit falls flat."[39] Peter Vincent from The Sydney Morning Herald called the video "unoriginal", citing the many popular culture references, and doubted Swift's success in transforming her image to pop.[24] Media professor Maryn Wilkinson noted "Shake It Off" as a representation of Swift's "zany" persona during the 1989 era.[note 1] Wilkinson noted that as Swift had been associated with a hardworking and authentic persona through her country songs, her venture to "artificial, manufactured" pop required intricate maneuvering to retain her sense of authenticity.[88] As observed by Wilkinson, in the video, after failing every dance routine, Swift laughs at herself implying that she will never "fit in" to "any commercially viable image, and prefers to embrace her natural zany state instead".[89] In doing so, Swift reminded the audience of her authenticity underneath "the artificial manufacture of pop performances".[89]

"Shake It Off" attracted allegations of racism and cultural appropriation for perpetuating African American stereotypes such as twerking and breakdancing. Its release coinciding with the race relation debates revolving the Ferguson unrest was also met with criticism.[90][91] Analyzing the video's supposedly "racializing surveillance" in a post-racial context, communications professor Rachel Dubrofsky noted the difference between Swift's depiction of conventionally white dance moves—such as ballet and cheerleading; and conventionally black dance moves—breakdancing and twerking.[note 2] She argued that while Swift's outfits and demeanor when she performs ballet or cheerleading fit her "naturally", she "does not easily embody the break-dancer's body nor does the style of dress [while twerking] fit her seamlessly".[83] Dubrofsky summarized the video as Swift's statement of her white authenticity: "I'm so white, you know it, I know it, which makes it so funny when I try to dance like a person of color."[93]

The Washington Post noted the video's depiction of dance moves associated with people of color, such as twerking, was another case of an ongoing debate about white pop singers embracing black culture.[94] Romanek defended his work: "We simply choose styles of dance that we thought would be popular and amusing ... If you look at [the video] carefully, it's a massively inclusive piece. And ... it's a satirical piece. It's playing with a whole range of music video tropes and cliches and stereotypes".[85][95]
Accolades

"Shake It Off" appeared on many publications' lists of the best songs of 2014. It featured in the top ten on lists by Time Out (third),[96] PopMatters (fourth),[97] The Village Voice's Pazz & Jop critics' poll (fourth),[98] and Consequence (eighth).[99] The track featured on 2014 year-end lists by Drowned in Sound (14th),[100] Dagsavisen (16th),[101] and NME (27th).[102] It was ranked by NME and Consequence as the 19th and 38th best song of the 2010s decade, respectively.[103][104] USA Today listed "Shake It Off" as one of the ten songs that defined the 2010s.[105]

"Shake It Off" has received many industry awards and nominations. It was honored by the 2015 Nashville Songwriters Association International, where Swift was the Songwriter of the Year.[106][107] The song received an award at the 2016 BMI Pop Awards, where Swift also earned the distinction of Songwriter of the Year.[108] At the 57th Annual Grammy Awards in 2015, "Shake It Off" was nominated in three categories: Record of the Year, Song of the Year, (both categories lost to "Stay with Me" by Sam Smith) and Best Pop Solo Performance but lost to "Happy" by Pharrell Williams.[109]

At the 2015 Billboard Music Awards, "Shake It Off" received three nominations, winning Top Streaming Song (Video).[110] "Shake It Off" won Song of the Year at the 2015 iHeartRadio Music Awards,[111] Favorite International Video at the 2015 Myx Music Awards (Philippines),[112] and Favorite Song at the 2015 People's Choice Awards.[113] The song was nominated for the Nickelodeon Kids' Choice Awards,[114] Teen Choice Awards,[115] Rockbjörnen Awards (Sweden),[116] Radio Disney Music Awards,[117] and Los Premios 40 Principales (Spain).[118]
Live performances
Taylor Swift wearing a pink crop top and skirt while performing on the 1989 World Tour
Swift performing "Shake It Off" on the 1989 World Tour (2015)

Swift premiered "Shake It Off" on television at the 2014 MTV Video Music Awards on August 24, 2014, with Kiki Wong on drums.[119][120] She performed the song at the German Radio Awards on September 4.[121] As part of promotion of 1989, she performed the song on television shows including The X Factor UK on October 12,[122] The X Factor Australia on October 20,[123] Jimmy Kimmel Live! on October 23,[124] and Good Morning America on October 29.[125] On October 27, 2014, the day of 1989's release, she performed the song as part of a mini-concert titled the "1989 Secret Sessions", live broadcast by Yahoo! and iHeartRadio.[126] She also played "Shake It Off" on music festivals including the iHeartRadio Music Festival on September 19,[127] the We Can Survive benefit concert at the Hollywood Bowl on October 24,[128] and the Jingle Ball Tour 2014 on December 5.[129] At the after party for the 40th anniversary of Saturday Night Live, Swift performed the song in an impromptu performance with Jimmy Fallon on backing vocals and Paul McCartney on backing vocals and bass guitar.[130]

On April 23, 2019, she performed an acoustic version of the song at the Time 100 Gala, where she was honored as one of the "100 most influential" people of the year.[131] She again performed the song on the finale of the eighth season of The Voice France on May 25,[132] on the Wango Tango festival on June 1,[133] at the City of Lover one-off concert in Paris on September 9,[134] and at the We Can Survive charity concert in Los Angeles on October 19, 2019.[135] At the 2019 American Music Awards, where she was honored as the Artist of the Decade, Swift performed "Shake It Off" as part of a medley of her hits. Halsey and Cabello joined Swift onstage during the song.[136] She again performed the song at Capital FM's Jingle Bell Ball 2019 in London[137] and at iHeartRadio Z100's Jingle Ball in New York City.[138]

"Shake It Off" was included on the set lists on three of Swift's world tours—the 1989 World Tour (2015), where the song was the final number,[139] the Reputation Stadium Tour (2018), where she performed the song with Camila Cabello and Charli XCX as supporting acts,[140] and the Eras Tour (2023–2024).[141]
Controversies
2014 Triple J Hottest 100

Following a January 13, 2015, BuzzFeed article titled "Why Isn't Everyone Voting For 'Shake It Off' In The Hottest 100?", the #Tay4Hottest100 hashtag campaign on social media emerged during the voting period for the Triple J Hottest 100, an annual poll selecting the 100 most prominent songs by the Australian radio station Triple J.[142] The social media posts tagged with #Tay4Hottest100 overwhelmed those mentioning the official contenders.[143] The campaign led to a significant amount of media debate over the merits of Swift's inclusion in the poll.[144][145] One criterion for eligibility was being played on air by Triple J at least once in 2014; Swift's "Shake It Off" did not receive airplay, but a cover of the song by the folk group Milky Chance did.[146] Critics of the campaign argued that the Hottest 100 was a platform for up-and-rising, non-mainstream artists,[147][148] but defenders criticized Triple J for embodying cultural elitism and sexism, citing how the radio prioritized "masculine 'rockist'" and "alternative" artists.[144][148] Guardian Australia's Elle Hunt wrote: "[The] virulent response to #Tay4Hottest100 has revealed the persistence of a dichotomy I'd thought we'd thrown out long ago: that of high art versus low."[149]

On January 20, 2015, Guardian Australia submitted a freedom of information request to the ABC in regard to the station's response to the campaign and the eligibility of "Shake It Off" for the Hottest 100 contest.[150] Triple J's manager Chris Scaddan told the website Tone Deaf: "We don't comment on voting campaigns while Hottest 100 voting is open. It draws attention to them and may influence the results of the poll."[151] "Shake It Off" was eventually disqualified by Triple J on January 26, 2015;[152] in the announcement, Triple J acknowledged Swift's music and career but highlighted that her entry—which had not received airtime—would not reflect their spirit.[144] They subsequently introduced two new rules that prohibited "trolling the poll"-type campaigns for the proceeding Hottest 100 polls.[153] The communications scholar Glen Fuller described the #Tay4Hottest100 campaign as an example of "connective action" in the age of social media.[148] As noted by Fuller, the emergence of personalized "action frames" expressing personal viewpoints intertwining with a larger framework of information created by media publications resulted in fragmented arguments that failed to result in a definite outcome.[154]
Lawsuits

In November 2015, Jessie Braham, an R&B singer known by the stage name Jesse Graham, claimed that Swift plagiarized his 2013 song "Haters Gonna Hate", citing his lyrics: "Haters gone hate, playas gone play. Watch out for them fakers, they'll fake you everyday."[155][156] In his lawsuit, he alleged that 92% of Swift's "Shake It Off" came from his song and demanded $42 million in damages from Swift and the distributor Sony.[157] On November 12, 2015, the lawsuit was dismissed by U.S. District Court Judge Gail Standish, who ruled that Braham did not have enough factual evidence but could file a new complaint "if his lawsuit deficiencies are corrected".[157] Standish quoted lyrics from Swift's songs "We Are Never Ever Getting Back Together", "Bad Blood", "Blank Space" and "Shake It Off":

    At present, the Court is not saying that Braham can never, ever, ever get his case back in court. But, for now, we have got problems, and the Court is not sure Braham can solve them. As currently drafted, the Complaint has a blank space – one that requires Braham to do more than write his name. And, upon consideration of the Court's explanation ... Braham may discover that mere pleading BandAids will not fix the bullet holes in his case. At least for the moment, Defendants have shaken off this lawsuit.[158]

In September 2017, the songwriters Sean "Sep" Hall and Nate Butler sued Swift for copyright infringement. They alleged that the lyrics of "Shake It Off" plagiarized those of "Playas Gon' Play" (2001), a song they wrote for the girl group 3LW, citing their lyrics: "Playas they gon' play, and haters they gonna hate / Ballers they gon' ball, shot callers they gonna call."[159] U.S. District Judge Michael W. Fitzgerald, in February 2018, dismissed the case on the grounds that the lyrics in question were too "banal" to be copyrighted;[160] but U.S. Circuit Judges John B. Owens, Andrew D. Hurwitz, and Kenneth K. Lee of the U.S. Court of Appeals for the Ninth Circuit, in October 2019, reversed the ruling, holding that the district court had "constituted itself as the final judge of the worth of an expressive work", and sent the case back to the district court.[161]

Swift's legal team filed new documents for dismissal of the suit in July 2020,[162] and in July 2021, filed for a summary judgment, arguing that the discovery phase of the lawsuit has turned up evidence in their favor.[163] On December 9, 2021, Fitzgerald refused Swift's request for a summary judgement.[164] Swift's legal team filed a second motion to dismiss the case on December 23, claiming the Fitzgerald's ruling was "unprecedented and cheats the public domain" if the plaintiffs could sue everyone who uses the phrases in any songwriting, singing or says it publicly.[165] On January 14, 2022, Hall and Butler's legal team filed a response stating, "The rules simply do not provide defendants with vehicles for rehashing old arguments and are not intended to give an unhappy litigant one additional chance to sway the judge."[166] On December 12, 2022, the lawsuit was dropped with no final verdict.[167]
Cover versions and usage in media
A portrait of Witherspoon
A portrait of Nyong'o
"Shake It Off" was covered by Reese Witherspoon (left) in the film Sing (2016) and Lupita Nyong'o in Little Monsters (2019).

Many musicians have covered "Shake It Off". Labrinth covered it at BBC Radio 1's Live Lounge September 20, 2014,[168] and Charli XCX performed a punk rock–inspired version at BBC Radio 1's Live Lounge on February 10, 2015;[169] the latter version was nominated for Best Cover Song at the 2015 mtvU Woodie Awards.[170] Ryan Adams covered "Shake It Off" for his track-by-track interpretation of Swift's 1989, released in September 2015. Adams said that Swift's 1989 helped him cope with emotional hardships and that he wanted to interpret the songs from his perspective "like it was Bruce Springsteen's Nebraska".[171] His version of "Shake It Off" incorporates acoustic instruments and a thumping drum line that critics found reminiscent of the drums on Bruce Springsteen's 1985 song "I'm on Fire".[172][173][174] Coldplay covered "Shake It Off" during their Music of the Spheres World Tour shows at Ernst-Happel-Stadion in Vienna, Austria, on August 22 and 24, 2024, as a tribute to the three cancelled shows of Swift's the Eras Tour following the uncovering of a terror plot.[175]

"Shake It Off" has been parodied and adapted into other mediums. In an April 2015 episode of Lip Sync Battle, the actor Dwayne Johnson lip synced to "Shake It Off" and Bee Gees' "Stayin' Alive" (1977) in a battle against Jimmy Fallon, and won.[176] The title of "Chris Has Got a Date, Date, Date, Date, Date", a Family Guy episode featuring a fictionalized character of Swift aired on November 6, 2016, is a pun on the lyrics of "Shake It Off".[177] The actress Reese Witherspoon and the comedian Nick Kroll performed an EDM–influenced version for the soundtrack to the musical animated film Sing (2016).[178] "Shake It Off" was also sung by the Mexican–Kenyan actress Lupita Nyong'o on a ukulele in the comedy film Little Monsters (2019).[179] A cover by the cast of the 2020 television series Zoey's Extraordinary Playlist was featured in the final episode of its second season.[180] "Weird Al" Yankovic covered "Shake It Off" as the final song of his 2024 polka medley "Polkamania!".[181]
Personnel

Credits are adapted from the liner notes of 1989.[10]

    Taylor Swift – vocals, background vocals, songwriter, clapping, shouts
    Cory Bice – assistant recording
    Tom Coyne – mastering
    Serban Ghenea – mixing
    John Hanes – engineering for mix
    Sam Holland – recording
    Michael Ilbert – recording
    Jonas Lindeborg – trumpet
    Max Martin – producer, songwriter, keyboard, programming, claps, shouts
    Shellback – producer, songwriter, acoustic guitar, bass guitar, keyboard, background vocals, drums, programming, claps, shouts, percussion
    Jonas Thander – baritone saxophone
    Magnus Wiklund – trombone

Charts
Weekly charts
2014–2015 weekly chart performance of "Shake It Off" Chart (2014–2015) 	Peak
position
Australia (ARIA)[182] 	1
Austria (Ö3 Austria Top 40)[183] 	6
Belgium (Ultratop 50 Flanders)[184] 	17
Belgium (Ultratop 50 Wallonia)[185] 	14
Brazil (Billboard Hot 100)[186] 	56
Canada (Canadian Hot 100)[187] 	1
Canada AC (Billboard)[188] 	1
Canada CHR/Top 40 (Billboard)[189] 	1
Canada Hot AC (Billboard)[190] 	1
CIS Airplay (TopHit)[191] 	143
Czech Republic (Rádio – Top 100)[192] 	3
Czech Republic (Singles Digitál Top 100)[193] 	1
Denmark (Tracklisten)[73] 	4
Denmark Airplay (Tracklisten)[194] 	1
Euro Digital Song Sales (Billboard)[195] 	2
Finland (Suomen virallinen lista)[196] 	6
France (SNEP)[197] 	6
Germany (GfK)[75] 	5
Greece Digital Song Sales (Billboard)[198] 	3
Hungary (Single Top 40)[199] 	1
Ireland (IRMA)[70] 	3
Israel (Media Forest)[74] 	4
Italy (FIMI)[200] 	10
Japan (Japan Hot 100)[201] 	4
Japan Adult Contemporary (Billboard)[202] 	1
Lebanon (Lebanese Top 20)[203] 	15
Luxembourg Digital Songs (Billboard)[204] 	8
Mexico Airplay (Billboard)[205] 	1
Netherlands (Dutch Top 40)[76] 	5
Netherlands (Single Top 100)[206] 	7
New Zealand (Recorded Music NZ)[63] 	1
Norway (VG-lista)[71] 	3
Poland (Polish Airplay Top 100)[68] 	1
Portugal Digital Songs (Billboard)[207] 	3
Romania (Airplay 100)[208] 	71
Scotland (OCC)[209] 	2
Slovakia (Rádio Top 100)[210] 	4
Slovakia (Singles Digitál Top 100)[211] 	1
Slovenia (SloTop50)[212] 	8
South Africa (EMA)[213] 	2
South Korean International Singles (Gaon)[214] 	19
Spain (PROMUSICAE)[69] 	2
Sweden (Sverigetopplistan)[72] 	3
Switzerland (Schweizer Hitparade)[215] 	7
UK Singles (OCC)[216] 	2
US Billboard Hot 100[217] 	1
US Adult Contemporary (Billboard)[55] 	1
US Adult Pop Airplay (Billboard)[218] 	1
US Country Airplay (Billboard)[219] 	58
US Dance Club Songs (Billboard)[220] 	17
US Latin Airplay (Billboard)[221] 	48
US Pop Airplay (Billboard)[222] 	1
US Rhythmic (Billboard)[223] 	17
2023–2024 weekly chart performance of "Shake It Off" Chart (2023-2024) 	Peak
position
Global 200 (Billboard)[224] 	78
Portugal (AFP)[225] 	38
Singapore (RIAS)[226] 	20
	
Year-end charts
2014 year-end charts for "Shake It Off" Chart (2014) 	Position
Australia (ARIA)[227] 	3
Austria (Ö3 Austria Top 40)[228] 	58
Canada (Canadian Hot 100)[229] 	9
France (SNEP)[230] 	101
Germany (Official German Charts)[231] 	39
Hungary (Single Top 40)[232] 	48
Ireland (IRMA)[233] 	12
Israel (Media Forest)[234] 	48
Japan (Japan Hot 100)[235] 	34
Netherlands (Dutch Top 40)[236] 	55
Netherlands (Single Top 100)[237] 	77
New Zealand (Recorded Music NZ)[238] 	3
Poland (ZPAV)[239] 	45
Spain (PROMUSICAE)[240] 	42
Switzerland (Schweizer Hitparade)[241] 	49
UK Singles (Official Charts Company)[242] 	14
US Billboard Hot 100[243] 	13
US Adult Contemporary (Billboard)[244] 	20
US Adult Top 40 (Billboard)[245] 	19
US Mainstream Top 40 (Billboard)[246] 	19
2015 year-end charts for "Shake It Off" Chart (2015) 	Position
Australia (ARIA)[247] 	49
Belgium (Ultratop Wallonia)[248] 	71
Canada (Canadian Hot 100)[249] 	17
France (SNEP)[250] 	116
Hungary (Single Top 40)[251] 	54
Japan (Japan Hot 100)[252] 	12
Slovenia (SloTop50)[253] 	31
US Billboard Hot 100[254] 	18
US Adult Contemporary (Billboard)[255] 	15
US Adult Top 40 (Billboard)[256] 	43
2016 year-end chart for "Shake It Off" Chart (2016) 	Position
Japan (Japan Hot 100)[257] 	62
2017 year-end chart for "Shake It Off" Chart (2017) 	Position
Japan (Japan Hot 100)[258] 	100
2023 year-end charts for "Shake It Off" Chart (2023) 	Position
Australia (ARIA)[259] 	81
Global 200 (Billboard)[260] 	143
Decade-end charts
2010s decade-end charts for "Shake It Off" Chart (2010–2019) 	Position
Australia (ARIA)[261] 	23
UK Singles (Official Charts Company)[262] 	76
US Billboard Hot 100[263] 	34
US Digital Songs (Billboard)[264] 	41
US Streaming Songs (Billboard)[265] 	26
All-time charts
All-time charts for "Shake It Off" Chart (1958–2018) 	Position
US Billboard Hot 100 (Female)[266] 	46
US Billboard Hot 100[267] 	139
US Adult Top 40 (Billboard)[268] 	37

Certifications
Certifications for "Shake It Off" Region 	Certification 	Certified units/sales
Australia (ARIA)[61] 	18× Platinum 	1,260,000‡
Austria (IFPI Austria)[78] 	2× Platinum 	60,000*
Belgium (BEA)[269] 	Gold 	20,000‡
Brazil (Pro-Música Brasil)[77] 	3× Diamond 	750,000‡
Canada (Music Canada)[62] 	6× Platinum 	480,000*
Denmark (IFPI Danmark)[270] 	Platinum 	90,000‡
Germany (BVMI)[271] 	3× Gold 	900,000‡
Italy (FIMI)[79] 	2× Platinum 	200,000‡
Japan (RIAJ)[67] 	3× Platinum 	750,000*
Mexico (AMPROFON)[272] 	Gold 	30,000*
New Zealand (RMNZ)[273] 	6× Platinum 	180,000‡
Norway (IFPI Norway)[80] 	2× Platinum 	120,000‡
Portugal (AFP)[274] 	Gold 	10,000‡
Spain (PROMUSICAE)[81] 	2× Platinum 	120,000‡
Sweden (GLF)[275] 	Platinum 	40,000‡
Switzerland (IFPI Switzerland)[276] 	Platinum 	30,000‡
United Kingdom (BPI)[65] 	5× Platinum 	3,000,000‡
United States (RIAA)[59] 	Diamond 	10,000,000‡
Streaming
Denmark (IFPI Danmark)[277] 	Gold 	1,300,000†
Japan (RIAJ)[278] 	Platinum 	100,000,000†

* Sales figures based on certification alone.
‡ Sales+streaming figures based on certification alone.
† Streaming-only figures based on certification alone.
Release history
Release dates and formats for "Shake It Off" Region 	Date 	Format 	Label(s) 	Ref.
Various 	August 19, 2014 	Digital download 	Big Machine 	[31]
United States 	Contemporary hit radio 	

    Big MachineRepublic

	[32]
Italy 	August 29, 2014 	Radio airplay 	Universal 	[35]
Various 	September 11, 2014 	CD single 	Big Machine 	[33]
Germany 	October 10, 2014 	Universal 	[36]
"Shake It Off (Taylor's Version)"
"Shake It Off (Taylor's Version)"
Song by Taylor Swift
from the album 1989 (Taylor's Version)
Released	October 27, 2023
Studio	Prime Recording (Nashville)
Length	3:39
Label	Republic
Songwriter(s)	

    Taylor Swift Max Martin Shellback

Producer(s)	

    Taylor Swift Christopher Rowe

Lyric video
"Shake It Off (Taylor's Version)" on YouTube

After signing a new contract with Republic Records, Swift began re-recording her first six studio albums in November 2020.[279] The decision followed a public 2019 dispute between Swift and talent manager Scooter Braun, who acquired Big Machine Records, including the masters of Swift's albums which the label had released.[280][281] By re-recording the albums, Swift had full ownership of the new masters, which enabled her to control the licensing of her songs for commercial use. In doing so, she hoped that the re-recorded songs would substitute the Big Machine–owned masters.[282]

The re-recording of "Shake It Off", subtitled "Taylor's Version", was released as part of 1989's re-recording, 1989 (Taylor's Version), on October 27, 2023.[283] Swift produced "Shake It Off (Taylor's Version)" with Christopher Rowe, who had produced her previous re-recordings.[284] The track was engineered by Derek Garten and Lowell Reynolds at Prime Recording Studio in Nashville, Tennessee; mixed by Ghenea at MixStar Studios in Virginia Beach, Virginia; and mastered by Randy Merrill at Sterling Sound in Edgewater, New Jersey. Rowe and Sam Holland recorded Swift's vocals at Conway Recording Studios in Los Angeles and Kitty Committee Studio in New York.[285]
Personnel

Credits are adapted from the liner notes of 1989 (Taylor's Version).[285]

Technical

    Taylor Swift – producer
    Bryce Bordone – engineer for mix
    Mattias Bylund – horn recording, horn editing
    Derek Garten – engineering, additional programming, editing
    Serban Ghenea – mixing
    Sam Holland – vocals recording
    Lowell Reynolds – engineering, additional programming, editing
    Christopher Rowe – vocals recording, producer

Musicians

    Taylor Swift – vocals, background vocals, songwriter
    Robert Allen – foot stomps, handclaps, background vocals
    Max Bernstein – synth horns
    Matt Billingslea – percussion
    Janne Bjerger – trumpet
    Mattias Bylund – synth horns, conducting
    Wojtek Goral – alto saxophone, baritone saxophone
    Amos Heller – bass
    Peter Noos Johansson – trombone, tuba
    Magnus Johansson – trumpet
    Tomas Jönsson – baritone saxophone, tenor saxophone
    Max Martin – songwriter
    Mike Meadows – synthesizer, background vocals
    Christopher Rowe – trumpet, background vocals
    Paul Sidoti – electric guitar, background vocals
    Shellback – songwriter, drums, laser harp

Charts
Chart performance for "Shake It Off (Taylor's Version)" Chart (2023) 	Peak
position
Australia (ARIA)[286] 	18
Canada (Canadian Hot 100)[287] 	24
Global 200 (Billboard)[288] 	21
Greece International (IFPI)[289] 	36
New Zealand (Recorded Music NZ)[290] 	27
Philippines (Billboard)[291] 	22
Sweden Heatseeker (Sverigetopplistan)[292] 	4
UK Singles Downloads (OCC)[293] 	46
UK Singles Sales (OCC)[294] 	53
UK Streaming (OCC)[295] 	26
US Billboard Hot 100[296] 	28
Vietnam (Vietnam Hot 100)[297] 	94
Certifications
Certifications for "Shake It Off (Taylor's Version)" Region 	Certification 	Certified units/sales
Australia (ARIA)[61] 	Gold 	35,000‡
Brazil (Pro-Música Brasil)[298] 	Gold 	20,000‡

‡ Sales+streaming figures based on certification alone.
See also

    List of highest-certified singles in Australia
    List of Billboard Hot 100 number ones singles of 2014
    List of Billboard Adult Contemporary number ones of 2014
    List of Billboard Adult Contemporary number ones of 2015
    List of Canadian Hot 100 number-one singles of 2014
    List of number-one digital songs of 2014 (U.S.)
    List of number-one singles of 2014 (Australia)
    List of number-one singles from the 2010s (New Zealand)
    List of most-viewed YouTube videos
    List of most-liked YouTube videos

Footnotes

Wilkinson used "zany" to describe Swift as "a figure who emphasises the pop 'performance' as one of hard work instead, because she exposed its construction as one that does not come 'naturally'".[87]

    Dubrofsky, citing Simone Browne, describes "racializing surveillance" as "a technology of social control where surveillance practices, policies, and performances concern the production of norms pertaining to race and exercise a power to define what is in or out of place."[92]

References

Caulfield, Keith (October 30, 2012). "Taylor Swift's Red Sells 1.21 Million; Biggest Sales Week for an Album Since 2002". Billboard. Archived from the original on February 1, 2013. Retrieved February 4, 2019.
McNutt 2020, p. 77.
McNutt 2020, pp. 77–78.
Doyle, Patrick (July 15, 2013). "Taylor Swift: 'Floodgates Open' for Next Album". Rolling Stone. Archived from the original on February 25, 2019. Retrieved February 25, 2019.
Vinson, Christina (September 8, 2014). "Taylor Swift on Turning Away from Country Music on 1989". Taste of Country. Archived from the original on June 30, 2020. Retrieved August 7, 2020.
McNutt 2020, p. 78.
Talbott, Chris (October 12, 2013). "Taylor Swift Talks Next Album, CMAs and Ed Sheeran". Associated Press. Archived from the original on October 26, 2013. Retrieved September 8, 2014.
Eells, Josh (September 8, 2014). "Cover Story: The Reinvention of Taylor Swift". Rolling Stone. Archived from the original on August 16, 2018. Retrieved February 6, 2019.
Lipshutz, Jason (August 19, 2014). "Taylor Swift Is Going Pop. And That's a Good Thing". Billboard. Archived from the original on November 17, 2021. Retrieved December 7, 2020.
Swift, Taylor (2014). 1989 (CD liner notes). Big Machine Records. BMRBD0500A.
Millman, Ethan (August 9, 2022). "Taylor Swift On 'Shake It Off' Lawsuit: 'The Lyrics Were Written Entirely By Me'". Rolling Stone. Archived from the original on August 10, 2022. Retrieved August 12, 2022.
Savage, Mark (October 27, 2023). "Taylor Swift's Biggest Album 1989 Returns with New Tracks From the Vault". BBC. Archived from the original on February 12, 2024. Retrieved November 15, 2023.
Erlewine, Stephen Thomas. "Taylor Swift – Artist Biography". AllMusic. Archived from the original on April 5, 2015. Retrieved June 15, 2015.
Wood, Mikael (August 18, 2014). "Listen: Taylor Swift releases 'Shake It Off,' from new album '1989'". Los Angeles Times. Archived from the original on August 19, 2014. Retrieved August 19, 2014.
Zollo, Paul (February 12, 2015). "The Oral History of Taylor Swift's 1989". The Recording Academy. Archived from the original on April 4, 2016. Retrieved February 27, 2019 – via Cuepoint.
Nobile 2015, p. 200.
"Taylor Swift's 'Shake It Off'". Drowned in Sound. August 19, 2014. Archived from the original on August 8, 2020. Retrieved December 7, 2020.
Terich, Jeff (August 19, 2014). "Track Review: Taylor Swift, 'Shake It Off'". American Songwriter. Archived from the original on December 13, 2020. Retrieved December 7, 2020.
Feeney, Nolan (August 18, 2014). "Watch Taylor Swift Show Off Her Dance Moves in New 'Shake It Off' Video". Time. Archived from the original on January 30, 2019. Retrieved August 19, 2014.
Kreps, Daniel (August 18, 2014). "Taylor Swift Dismisses the Haters in New Song 'Shake It Off'". Rolling Stone. Archived from the original on August 19, 2014. Retrieved August 19, 2014.
Block, Melissa (October 31, 2014). "'Anything That Connects': A Conversation With Taylor Swift" (Audio upload and transcript). NPR. Archived from the original on February 6, 2015. Retrieved January 30, 2015.
Ezell, Brice; Sawdey, Evan (September 21, 2017). "The Flipside #7: Taylor Swift's '1989'". PopMatters. Archived from the original on December 13, 2020. Retrieved December 7, 2020.
Molanphy, Chris (August 29, 2014). "Why Is Taylor Swift's 'Shake It Off' No. 1?". Slate. Archived from the original on December 6, 2020. Retrieved December 7, 2020.
Vincent, Peter (August 19, 2014). "Taylor Swift Laughs Off Critics, But Can't Match Boy Bands". The Sydney Morning Herald. Archived from the original on October 30, 2020. Retrieved December 7, 2020.
Roberts, Randall (August 20, 2014). "Critic's Notebook: Taylor Swift's Catchy, Tone-Deaf 'Shake It Off'". Los Angeles Times. Archived from the original on November 1, 2020. Retrieved August 21, 2014.
Jones, Nate (August 13, 2020). "All 162 Taylor Swift Songs, Ranked". Vulture. Archived from the original on September 13, 2019. Retrieved December 8, 2020.
"Taylor Swift Trademarks 'Sick Beat'". BBC. January 29, 2015. Archived from the original on March 4, 2021. Retrieved January 22, 2021.
Strecker, Erin (August 7, 2014). "Taylor Swift Drops Two More Clues About New Music". Billboard. Archived from the original on August 10, 2014. Retrieved August 18, 2014.
Koerber, Brian (August 14, 2014). "Ew, Taylor Swift Plays 'Show and Tell' With Jimmy Fallon". Entertainment Weekly. Archived from the original on December 7, 2018. Retrieved December 8, 2020.
Payne, Chris (August 18, 2014). "Taylor Swift Reveals New Album Title, Release Date & 'Shake It Off' Video". Billboard. Archived from the original on August 21, 2014. Retrieved August 19, 2014.
"Shake It Off (2014)". 7digital. Archived from the original on September 25, 2014. Retrieved September 25, 2014.
"Top 40/M Future Releases". All Access Music Group. Archived from the original on August 19, 2014. Retrieved August 19, 2014.
"Limited edition 'Shake It Off' Single CD". Taylorswift.com. Archived from the original on September 13, 2014. Retrieved September 12, 2014.
"BBC Radio 1 Playlist". BBC Radio 1. Archived from the original on August 26, 2014. Retrieved August 25, 2014.
Mompellio, Gabriel (August 26, 2014). "Taylor Swift – Shake It Off (Radio Date: 29-08-2014)" (in Italian). Universal Music Group. Archived from the original on October 28, 2014. Retrieved December 12, 2020.
"Shake It Off (CD)" (in German). Universal Music Group. Archived from the original on May 19, 2021. Retrieved January 1, 2021.
Willis, Charlotte (August 20, 2014). "'Shake It Off', Taylor Swift's New Single Falls Flat in Reviews". News.com.au. Archived from the original on December 13, 2020. Retrieved December 7, 2020.
Vincent, Alice (August 18, 2014). "Taylor Swift's New Single 'Shake It Off' Shakes Up Pop Music". The Daily Telegraph. Archived from the original on November 27, 2015. Retrieved August 19, 2014.
Fitzpatrick, Molly (August 19, 2014). "Taylor Swift's 'Shake it Off' Video Falls Flat". The Guardian. Archived from the original on October 6, 2014. Retrieved August 19, 2014.
Lipshutz, Jason (August 18, 2014). "Taylor Swift's Shake It Off: Single Review". Billboard. Archived from the original on October 27, 2020. Retrieved August 19, 2014.
Petridis, Alexis (October 24, 2014). "Taylor Swift: 1989 Review – Leagues Ahead of the Teen-Pop Competition". The Guardian. Archived from the original on November 1, 2014. Retrieved December 9, 2020.
Unterberger, Andrew (October 28, 2014). "Taylor Swift Gets Clean, Hits Reset on New Album 1989". Spin. Archived from the original on November 19, 2018. Retrieved April 5, 2018.
Lipshutz, Jason (December 11, 2019). "Taylor Swift: Billboard's Woman of the Decade Cover Story". Billboard. Archived from the original on February 26, 2021. Retrieved February 21, 2021.
Mylrea, Hannah (September 8, 2020). "Every Taylor Swift Song Ranked In Order of Greatness". NME. Archived from the original on September 17, 2020. Retrieved November 26, 2020.
Sheffield, Rob (November 24, 2020). "All 173 of Taylor Swift's Songs, Ranked". Rolling Stone. Archived from the original on December 13, 2020. Retrieved December 9, 2020.
Petridis, Alexis (April 26, 2019). "Taylor Swift's Singles – Ranked!". The Guardian. Archived from the original on April 27, 2019. Retrieved December 9, 2020.
Song, Jane (February 11, 2020). "All 158 Taylor Swift Songs, Ranked". Paste. Archived from the original on April 13, 2020. Retrieved December 9, 2020.
Trust, Gary (August 19, 2014). "Taylor Swift Turns Radio on With 'Shake It Off'". Billboard. Archived from the original on March 1, 2015. Retrieved August 20, 2014.
Trust, Gary (August 20, 2014). "Ariana Grande, Iggy Azalea Triple Up in Hot 100's Top 10, MAGIC! Still No. 1". Billboard. Archived from the original on November 8, 2015. Retrieved August 20, 2014.
Trust, Gary (August 25, 2014). "Taylor Swift's 'Shake It Off' Makes Record Start at Radio". Billboard. Archived from the original on November 12, 2020. Retrieved August 26, 2014.
Trust, Gary; Asker, Jim (November 17, 2017). "Taylor Swift's 'New Year's Day' Goes to Country Radio: Is Country Ready for It?". Billboard. Archived from the original on April 23, 2018. Retrieved March 19, 2017.
Trust, Gary (August 27, 2014). "Taylor Swift's 'Shake It Off' Debuts At No. 1 On Hot 100". Billboard. Archived from the original on January 23, 2015. Retrieved January 22, 2015.
Trust, Gary (November 5, 2014). "Taylor Swift's 'Shake It Off' Returns to No. 1 on Hot 100". Billboard. Archived from the original on November 15, 2020. Retrieved December 4, 2020.
Trust, Gary (November 12, 2014). "Taylor Swift Still No. 1 on Hot 100, Ariana Grande & The Weeknd Hit Top 10". Billboard. Archived from the original on October 2, 2020. Retrieved December 9, 2020.
"Taylor Swift Chart History (Adult Contemporary)". Billboard. Retrieved September 26, 2014.
"Decade-End Report" (PDF). Nielsen SoundScan. p. 39. Archived from the original (PDF) on January 11, 2020. Retrieved January 11, 2020.
Trust, Gary (February 22, 2024). "Taylor Swift's 50 Biggest Billboard Hot 100 Hits". Billboard. Archived from the original on March 16, 2024. Retrieved September 11, 2024.
"Taylor Swift Chart History (Hot 100)". Billboard. Archived from the original on November 19, 2021. Retrieved October 6, 2023.
"American single certifications – Taylor Swift – Shake It Off". Recording Industry Association of America. Retrieved June 2, 2020.
Ahlgrim, Callie (March 14, 2020). "There Are Only 34 Songs in History That Have Been Certified Diamond — Here They All Are". MSN. Archived from the original on October 23, 2020. Retrieved October 22, 2020.
"ARIA Charts – Accreditations – 2024 Singles" (PDF). Australian Recording Industry Association. Retrieved February 14, 2024.
"Canadian single certifications – Taylor Swift – Shake It Off". Music Canada. Retrieved March 13, 2015.
"Taylor Swift – Shake It Off". Top 40 Singles. Retrieved September 1, 2014.
Griffiths, George (November 14, 2022). "Taylor Swift's 'Shake It Off' Becomes an Official UK Million-Seller". Official Charts Company. Retrieved June 24, 2024.
"British single certifications – Taylor Swift – Shake It Off". British Phonographic Industry. Retrieved May 31, 2024.
Griffiths, George (April 23, 2024). "Taylor Swift's Official Top 40 Biggest Songs In the UK Revealed". Official Charts Company. Retrieved June 3, 2024.
"Japanese digital single certifications – Taylor Swift – Shake It Off" (in Japanese). Recording Industry Association of Japan. Retrieved January 18, 2021. Select 2017年2月 on the drop-down menu
"Listy bestsellerów, wyróżnienia :: Związek Producentów Audio-Video". Polish Airplay Top 100. Retrieved November 10, 2014.
"Taylor Swift – Shake It Off" Canciones Top 50. Retrieved August 28, 2014.
"The Irish Charts – Search Results – Shake It Off". Irish Singles Chart. Retrieved January 28, 2020.
"Taylor Swift – Shake It Off". VG-lista. Retrieved October 11, 2014.
"Taylor Swift – Shake It Off". Singles Top 100. Retrieved September 25, 2014.
"Taylor Swift – Shake It Off". Tracklisten. Retrieved September 19, 2014.
"Media Forest Week 41, 2014". Israeli Airplay Chart. Media Forest. Retrieved January 29, 2015.
"Taylor Swift – Shake It Off" (in German). GfK Entertainment charts. Retrieved August 26, 2014.
"Nederlandse Top 40 – Taylor Swift" (in Dutch). Dutch Top 40. Retrieved September 5, 2014.
"Brazilian single certifications – Taylor Swift – Shake It Off" (in Portuguese). Pro-Música Brasil. Retrieved March 22, 2024.
"Austrian single certifications – Taylor Swift – Shake It Off" (in German). IFPI Austria. Retrieved May 29, 2024.
"Italian single certifications – Taylor Swift – Shake It Off" (in Italian). Federazione Industria Musicale Italiana. Retrieved May 27, 2024.
"Norwegian single certifications – Taylor Swift – Shake It Off" (in Norwegian). IFPI Norway. Retrieved October 29, 2020.
"Spanish single certifications – Taylor Swift – Shake It Off". El portal de Música. Productores de Música de España. Retrieved April 4, 2024.
Cho, Diane (August 20, 2014). "A Breakdown of Every Cultural Reference in Taylor Swift's 'Shake It Off' Music Video". VH1. Archived from the original on August 21, 2014. Retrieved August 21, 2014.
Dubrofsky 2016, p. 192.
Michaels, Sean (August 19, 2014). "Taylor Swift Announces New Album Details and Single 'Shake It Off'". The Guardian. Archived from the original on August 21, 2014. Retrieved August 19, 2014.
Buchanan, Kyle (August 21, 2014). "Mark Romanek on Directing Taylor Swift's New Video 'Shake It Off'". Vulture. Archived from the original on September 6, 2014. Retrieved September 5, 2014.
Sacks, Ethan (August 18, 2014). "Taylor Swift Reveals New Album 1989, Video for First Single 'Shake It Off'". New York Daily News. Archived from the original on November 2, 2017. Retrieved August 19, 2014.
Wilkinson 2017, p. 441.
Wilkinson 2017, p. 442.
Wilkinson 2017, p. 443.
Hasty, Katie (August 18, 2014). "Taylor Swift's New Video 'Shake It Off' Features Twerking, Ballet, Haters". HitFix. Archived from the original on August 19, 2014. Retrieved August 19, 2014.
Smith, Troy (August 19, 2014). "Taylor Swift's 'Shake It Off' Video Sparks Accusations of Racism on Twitter". The Plain Dealer. Archived from the original on January 2, 2021. Retrieved August 19, 2014.
Dubrofsky 2016, p. 191.
Dubrofsky 2016, p. 193.
Butler, Bethonie; Stahl, Jessica (August 22, 2014). "Is Taylor Swift's 'Shake It Off' Music Video Offensive? That's What People Say, Mmm Mmm". The Washington Post. Archived from the original on January 2, 2021. Retrieved August 22, 2014.
Bueno, Antoinette (August 22, 2014). "Taylor Swift's 'Shake It Off' Video Director Hits Back at Racist Claims". Entertainment Tonight. Archived from the original on September 6, 2014. Retrieved September 5, 2014.
"The 50 best songs of 2014". Time Out. January 20, 2015. Archived from the original on November 15, 2021. Retrieved November 14, 2021.
"The 75 Best Songs of 2014". PopMatters. December 22, 2014. Archived from the original on December 22, 2014. Retrieved November 14, 2021.
"Pazz & Jop: 2014 Singles (All Votes)". The Village Voice. Archived from the original on February 3, 2015. Retrieved February 3, 2015.
"Top 50 Songs of 2014". Consequence. December 5, 2014. Archived from the original on January 22, 2015. Retrieved January 22, 2015.
Leedham, Robert (December 8, 2014). "Drowned in Sound's 40 Favourite Songs of 2014". Drowned in Sound. Archived from the original on October 31, 2020. Retrieved December 8, 2014.
"Her er listene over den beste musikken i 2014". Dagsavisen (in Norwegian). December 12, 2014. Archived from the original on November 15, 2021. Retrieved December 12, 2014.
Barker, Emily (November 24, 2014). "50 Top Tracks of 2014". NME. Archived from the original on July 18, 2021. Retrieved November 24, 2014.
"The Best Songs Of The Decade: The 2010s". NME. December 4, 2019. Archived from the original on December 4, 2019. Retrieved December 9, 2019.
"Top 100 Songs of the 2010s". Consequence of Sound. November 11, 2019. Archived from the original on November 11, 2019. Retrieved December 9, 2019.
Ryan, Patrick. "10 songs that defined the 2010s in music". USA Today. Archived from the original on December 18, 2019. Retrieved December 18, 2019.
"Taylor Swift Earns 7th Songwriter/Artist of the Year Award". Nashville Songwriters Association International. October 11, 2015. Archived from the original on October 13, 2015. Retrieved October 11, 2015.
"Ten Songs I Wish I'd Written". Nashville Songwriters Association International. October 11, 2015. Archived from the original on March 5, 2016. Retrieved March 5, 2016.
"BMI Honors Taylor Swift and Legendary Songwriting Duo Mann & Weil at the 64th Annual BMI Pop Awards". Broadcast Music, Inc. May 11, 2016. Archived from the original on May 27, 2016. Retrieved May 11, 2016.
"Grammys 2015: See the Full Winners List". Billboard. February 9, 2015. Archived from the original on February 10, 2015. Retrieved October 6, 2018.
"Billboard Music Awards 2015: See the Full Winners List". Billboard. May 17, 2015. Archived from the original on November 8, 2020. Retrieved November 8, 2020.
"2015 iHeartRadio Music Awards: Full Winners List". iHeartRadio. March 29, 2015. Archived from the original on March 31, 2015. Retrieved March 29, 2015.
"Myx Music Awards 2015 Winners". Myx. Archived from the original on January 2, 2021. Retrieved April 16, 2015.
"Nominees & Winners". People's Choice Awards. Archived from the original on March 6, 2016. Retrieved June 22, 2015.
"Kids' Choice Awards 2015: The Complete Winners List". The Hollywood Reporter. March 28, 2015. Archived from the original on February 9, 2016. Retrieved December 9, 2020.
"Winners of Teen Choice 2015 Announced". Teen Choice Awards. August 16, 2015. Archived from the original on August 18, 2015. Retrieved August 17, 2015.
"Nominerade & vinnare i Rockbjörnen 2015". Aftonbladet (in Swedish). August 11, 2015. Archived from the original on September 6, 2015. Retrieved August 29, 2015.
Gajewski, Ryan (April 26, 2015). "Taylor Swift, Ariana Grande Win Big at Radio Disney Music Awards". The Hollywood Reporter. Archived from the original on January 2, 2021. Retrieved April 25, 2020.
"Premios 40 Principales 2015" (in Spanish). Los 40 Principales. Archived from the original on October 13, 2015. Retrieved October 16, 2015.
Pilley, Max (June 14, 2024). "New Smashing Pumpkins guitarist Kiki Wong speaks out after first shows with band". NME. Retrieved October 12, 2024. "Kiki Wong is a Los Angeles rock guitarist who plays in Vigil of War. Among her claims to fame is playing drums for Taylor Swift's performance of "Shake It Off" at the 2014 MTV Video Music Awards."
Lee, Ashley (August 24, 2014). "VMAs: Taylor Swift Refuses to Jump in 'Shake It Off' Debut Performance". The Hollywood Reporter. Archived from the original on December 9, 2019. Retrieved December 9, 2019.
Chau, Thomas (September 4, 2014). "Taylor Swift Performs 'Shake It Off' at the 2014 German Radio Awards". PopCrush. Archived from the original on November 7, 2017. Retrieved September 4, 2014.
Gracie, Bianca (October 12, 2014). "Taylor Swift Performs 'Shake It Off' On 'The X Factor UK'". Idolator. Archived from the original on January 2, 2021. Retrieved December 15, 2020.
Daw, Robbie (October 20, 2014). "Taylor Swift Performs 'Shake It Off' On 'The X Factor' Australia". Idolator. Archived from the original on January 2, 2021. Retrieved December 14, 2020.
Dockterman, Eliana (October 24, 2014). "Watch Taylor Swift Perform 'Out of the Woods' on Jimmy Kimmel Live!". Time. Archived from the original on January 2, 2021. Retrieved December 12, 2015.
Lee, Ashley (October 30, 2014). "Taylor Swift Teases '1989' Tour During 'Good Morning America' Concert". Billboard. Archived from the original on January 11, 2019. Retrieved December 15, 2020.
Stutz, Colin (October 27, 2014). "Taylor Swift Live-Broadcasts Manhattan Rooftop Secret Session". The Hollywood Reporter. Archived from the original on October 31, 2014. Retrieved February 6, 2019.
Lipshutz, Jason (September 20, 2014). "Taylor Swift Shakes Off the 'Frenemies' During iHeartRadio Fest Performance: Watch". Billboard. Archived from the original on September 20, 2014. Retrieved December 9, 2019.
Edwards, Gavin (October 25, 2014). "Taylor Swift, Ariana Grande and Gwen Stefani Cover the Hollywood Bowl in Glitter". Rolling Stone. Archived from the original on February 28, 2019. Retrieved December 15, 2020.
Stutz, Colin (December 6, 2014). "Taylor Swift Beats Laryngitis, Sam Smith, Ariana Grande Shine at KIIS FM Jingle Ball". Billboard. Archived from the original on January 2, 2021. Retrieved December 15, 2020.
Stedman, Alex (February 16, 2015). "Taylor Swift, Paul McCartney, Prince Jam at 'SNL' Anniversary Special After-Party (Video)". Variety. Archived from the original on July 2, 2020. Retrieved June 30, 2020.
Brandle, Lars (April 24, 2019). "Taylor Swift Took Some of the World's Biggest Stars Down Memory Lane With This Performance". Billboard. Archived from the original on April 24, 2019. Retrieved April 24, 2019.
Iasimone, Ashley (May 25, 2019). "Taylor Swift Performs 'Shake It Off' & 'ME!' on 'The Voice' in France: Watch". Billboard. Archived from the original on June 12, 2019. Retrieved December 12, 2020.
Willman, Chris (June 2, 2019). "Taylor Swift Goes Full Rainbow for Pride Month at L.A. Wango Tango Show". Variety. Archived from the original on December 7, 2020. Retrieved December 16, 2020.
Mylrea, Hannah (September 10, 2019). "Taylor Swift's The City of Lover concert: a triumphant yet intimate celebration of her fans and career". NME. Archived from the original on September 16, 2019. Retrieved September 12, 2019.
Aniftos, Rania (October 20, 2019). "Taylor Swift, Billie Eilish & More Supported a Great Cause at 7th Annual We Can Survive Concert: Recap". Billboard. Archived from the original on October 22, 2019. Retrieved October 23, 2019.
Gracie, Bianca (November 24, 2019). "Taylor Swift Performs Major Medley Of Hits, Brings Out Surprise Guests For 'Shake It Off' at 2019 AMAs". Billboard. Archived from the original on November 26, 2019. Retrieved November 25, 2019.
Iasimone, Ashley (December 8, 2019). "Taylor Swift Performs 'Christmas Tree Farm' Live for the First Time at Capital FM's Jingle Bell Ball: Watch". Billboard. Archived from the original on December 9, 2019. Retrieved December 9, 2019.
Mastrogiannis, Nicole (December 14, 2019). "Taylor Swift Brings Holiday Cheer to Jingle Ball with "Christmas Tree Farm"". iHeartRadio. Archived from the original on December 14, 2019. Retrieved December 14, 2019.
Yahr, Emily (May 5, 2015). "Taylor Swift '1989' World Tour: Set list, costumes, the stage, the spectacle". The Washington Post. Archived from the original on October 12, 2018. Retrieved December 12, 2018.
Britton, Luke Morgan (May 9, 2018). "Taylor Swift joined by Camila Cabello and Charli XCX for 'Shake It Off' at 'Reputation' stadium tour opener". NME. Archived from the original on July 2, 2020. Retrieved June 30, 2020.
Shafer, Ellise (March 18, 2023). "Taylor Swift Eras Tour: The Full Setlist From Opening Night (Updating Live)". Variety. Archived from the original on March 18, 2023. Retrieved March 18, 2023.
Fuller 2017, p. 170.
Vincent, Peter (January 23, 2015). "Taylor Swift Campaign Has Swallowed Triple J Hottest 100". The Sydney Morning Herald. Archived from the original on January 25, 2015. Retrieved January 26, 2015.
Carniel, Jessica (April 30, 2016). "Triple J's Hottest" (PDF). University of Southern Queensland. p. 42. Archived from the original (PDF) on January 2, 2021. Retrieved December 12, 2020.
Fuller 2017, p. 167.
Tan, Monica; Hunt, Elle; Seidler, Jonno; Paddy, Chelsea (January 14, 2015). "Taylor Swift Fans Invade Triple J Hottest 100 – And Five Songs that Deserve No 1". The Guardian. Archived from the original on January 2, 2021. Retrieved January 2, 2021.
Vincent, Peter (January 20, 2015). "Triple J Hottest 100: Has Taylor Swift Been Dumped from Contention Due to KFC Ad?". The Sydney Morning Herald. Archived from the original on January 22, 2015. Retrieved January 23, 2015.
Fuller 2017, p. 168.
Hunt, Elle (January 19, 2015). "#Tay4Hottest100: Taylor Swift Campaign Shows It's Time for Triple J to Shake Off Cultural Elitism". The Guardian. Archived from the original on January 21, 2015. Retrieved January 21, 2015.
Hunt, Elle (January 20, 2015). "Taylor Swift Fans Have Spoken – But Will Triple J's Hottest 100 Listen?". The Guardian. Archived from the original on January 21, 2015. Retrieved January 21, 2015.
Harris, Joe (January 20, 2015). "The Guardian Says Triple J Are 'Sexist' For Ignoring Taylor Swift, & That's Just Dumb". Tone Deaf. Archived from the original on January 21, 2015. Retrieved January 21, 2015.
Adams, Cameron (January 26, 2015). "Taylor Swift Disqualified from Hottest 100". News.com.au. Archived from the original on January 26, 2015. Retrieved January 26, 2015.
Fuller 2017, p. 181.
Fuller 2017, pp. 177–179.
Plucinska, Joanna (November 2, 2015). "Taylor Swift Sued for $42 Million Over 'Shake It Off' Lyrics". Time. Archived from the original on September 10, 2016. Retrieved September 15, 2016.
Brodsky, Rachel (November 1, 2015). "Taylor Swift Is Being Sued for Allegedly Stealing 'Shake It Off' Lyrics". Spin. Archived from the original on September 12, 2016. Retrieved September 15, 2016.
Preuss, Andreas; Isidore, Chris; Burke, Samuel (November 12, 2015). "Taylor Swift Shakes Off Copyright Llawsuit". CNN. Archived from the original on June 20, 2017. Retrieved August 27, 2017.
"Judge 'shakes off' lawsuit against Taylor Swift ... by quoting Taylor Swift". USA Today. November 12, 2015. Archived from the original on July 9, 2017. Retrieved September 11, 2017.
Gaca, Anna (September 19, 2017). "Taylor Swift Hit With Copyright Lawsuit Over 'Shake It Off'". Spin. Archived from the original on January 30, 2018. Retrieved December 12, 2018.
Savage, Mark (February 14, 2018). "US Judge dismisses Taylor Swift 'haters' case as too 'banal'". BBC News. Archived from the original on December 7, 2018. Retrieved December 23, 2018.
Maddaus, Gene (October 28, 2019). "Appeals Court Revives 'Shake It Off' Lawsuit Against Taylor Swift". Variety. Archived from the original on October 29, 2019. Retrieved October 30, 2019.
Cooke, Chris (August 13, 2020). "Taylor Swift has another go at shaking off Shake It Off lyric theft action". Complete Music Update. Archived from the original on September 25, 2021. Retrieved September 25, 2021.
Cooke, Chris (July 21, 2021). "Taylor Swift seeks summary judgement in her ongoing Shake It Off lyric-theft dispute". Complete Music Update. Archived from the original on September 25, 2021. Retrieved September 25, 2021.
Donahue, Bill (December 9, 2021). "Taylor Swift Must Face Trial in 'Shake It Off' Copyright Lawsuit". Billboard. Archived from the original on December 10, 2021. Retrieved December 9, 2021.
Donahue, Bill (December 28, 2021). "Taylor Swift Files to Dismiss 'Shake It Off' Lawsuit After 'Unprecedented' Ruling". Billboard. Archived from the original on December 28, 2021. Retrieved July 11, 2022.
Donahue, Bill (January 14, 2022). "Taylor Swift's Accusers Say She Must Face 'Shake It Off' Trial, Even If She's 'Unhappy'". Billboard. Archived from the original on January 15, 2022. Retrieved July 11, 2022.
Donahue, Bill (December 12, 2022). "Taylor Swift Copyright Accusers Drop Lawsuit Over 'Shake It Off' After Five Years Of Litigation". Billboard. Archived from the original on December 12, 2022. Retrieved December 12, 2022.
"Labrinth Puts His Own Spin on Taylor Swift's 'Shake It Off'". MTV UK. September 22, 2014. Archived from the original on August 21, 2016. Retrieved January 13, 2016.
Blistein, Jon (February 10, 2015). "Charli XCX Turns Taylor Swift Punk With Raucous 'Shake It Off' Cover". Rolling Stone. Archived from the original on February 19, 2016. Retrieved January 13, 2016.
"Sam Smith, Charli XCX nominated for mtvU Woodie Awards". Business Standard. Press Trust of India. February 19, 2015. Archived from the original on January 2, 2021. Retrieved December 15, 2020.
Browne, David (September 21, 2015). "Ryan Adams on His Full-Album Cover of Taylor Swift's 1989". Rolling Stone. Archived from the original on September 25, 2023. Retrieved December 29, 2023.
Gracey, Oscar (September 21, 2015). "Ryan Adams' 1989: Track By Track". Yahoo!. Archived from the original on January 26, 2016. Retrieved January 7, 2016.
Winograd, Jeremy (October 21, 2015). "Review: Ryan Adams, 1989". Slant Magazine. Archived from the original on May 6, 2019. Retrieved August 17, 2020.
Zaleski, Annie (September 21, 2015). "Ryan Adams Transforms Taylor Swift's 1989 Into a Melancholy Masterpiece". The A.V. Club. Archived from the original on February 12, 2018. Retrieved February 12, 2018.
Jones, Damian (August 26, 2024). "Watch Coldplay Wrap Up their Vienna Tour with Cover of Taylor Swift's 'The 1'". NME. Archived from the original on August 27, 2024. Retrieved September 10, 2024.
Dornbush, Jonathon (April 3, 2015). "Dwayne 'The Rock' Johnson shakes it off to Taylor Swift, Bee Gees on 'Lip Sync Battle'". Entertainment Weekly. Archived from the original on October 23, 2020. Retrieved December 9, 2020.
"Taylor Swift's break-up songs subject of Family Guy". The Indian Express. November 10, 2016. Archived from the original on January 2, 2021. Retrieved December 15, 2020.
Daly, Rhian (December 11, 2016). "Reese Witherspoon Has Done an EDM Cover of Taylor Swift's 'Shake It Off'". NME. Archived from the original on December 14, 2016. Retrieved December 7, 2020.
Aniftos, Rania (January 30, 2019). "Lupita Nyong'o Says Taylor Swift's 'Shake It Off' Helped Her Get Out of a 'Funk'". Billboard. Archived from the original on October 9, 2019. Retrieved October 9, 2019.
Gelman, Vlada (May 14, 2021). "Zoey's Playlist Finale Sneak Peek: Time to 'Shake It Off' at Max's Goodbye Party". Yahoo! Entertainment. Archived from the original on May 14, 2021. Retrieved May 14, 2021.
Greene, Andy (July 19, 2024). "Hear 'Weird Al' Yankovic Take on Taylor Swift, Olivia Rodrigo, Billie Eilish on 'Polkamania!'". Rolling Stone. Archived from the original on August 3, 2024. Retrieved September 10, 2024.
"Taylor Swift – Shake It Off". ARIA Top 50 Singles. Retrieved September 1, 2014.
"Taylor Swift – Shake It Off" (in German). Ö3 Austria Top 40. Retrieved September 5, 2014.
"Taylor Swift – Shake It Off" (in Dutch). Ultratop 50. Retrieved September 5, 2014.
"Taylor Swift – Shake It Off" (in French). Ultratop 50. Retrieved August 23, 2014.
"Hot 100 Billboard Brasil – weekly". Billboard Brasil (in Brazilian Portuguese). November 8, 2014. Archived from the original on October 27, 2014. Retrieved September 29, 2014.
"Taylor Swift Chart History (Canadian Hot 100)". Billboard. Retrieved August 28, 2014.
"Taylor Swift Chart History (Canada AC)". Billboard. Retrieved October 16, 2014.
"Taylor Swift Chart History (Canada CHR/Top 40)". Billboard. Retrieved October 16, 2014.
"Taylor Swift Chart History (Canada Hot AC)". Billboard. Retrieved September 26, 2014.
Taylor Swift — Shake It Off. TopHit. Retrieved April 19, 2021.
"ČNS IFPI" (in Czech). Hitparáda – Radio Top 100 Oficiální. IFPI Czech Republic. Note: Select 47. týden 2014 in the date selector. Retrieved September 19, 2014.
"ČNS IFPI" (in Czech). Hitparáda – Digital Top 100 Oficiální. IFPI Czech Republic. Note: Select 43. týden 2014 in the date selector. Retrieved September 26, 2014.
"Taylor Swift – Shake It Off" (in Danish). Tracklisten. Retrieved September 20, 2016.
"Taylor Swift Chart History (Euro Digital Song Sales)". Billboard. Retrieved October 16, 2014.
"Taylor Swift: Shake It Off" (in Finnish). Musiikkituottajat. Retrieved September 9, 2014.
"Taylor Swift – Shake It Off" (in French). Les classement single. Retrieved January 31, 2015.
"Taylor Swift Chart History (Greece Digital Song Sales)". Billboard. Archived from the original on December 6, 2019. Retrieved November 9, 2021.
"Archívum – Slágerlisták – MAHASZ" (in Hungarian). Single (track) Top 40 lista. Magyar Hanglemezkiadók Szövetsége. Retrieved January 29, 2015.
"Taylor Swift – Shake It Off". Top Digital Download. Retrieved March 3, 2016.
"Taylor Swift Chart History (Japan Hot 100)". Billboard. Retrieved September 12, 2014.
"Japan Adult Contemporary Airplay Chart". Billboard Japan (in Japanese). Archived from the original on February 27, 2024. Retrieved October 31, 2023.
"Taylor Swift". The Official Lebanese Top 20. Archived from the original on September 17, 2016. Retrieved April 18, 2016.
"Taylor Swift Chart History (Luxembourg Digital Songs)". Billboard. Archived from the original on October 14, 2019. Retrieved December 30, 2023.
"Mexico Airplay". Billboard. Archived from the original on June 19, 2015. Retrieved November 1, 2014.
"Taylor Swift – Shake It Off" (in Dutch). Single Top 100. Retrieved August 29, 2014.
"Taylor Swift Chart History (Portugal Digital Songs)". Billboard. Archived from the original on October 14, 2019. Retrieved December 30, 2023.
"Airplay 100 – 2 noiembrie 2014" (in Romanian). Kiss FM. November 2, 2014. Archived from the original on March 28, 2020. Retrieved March 28, 2020.
"Official Scottish Singles Sales Chart Top 100". Official Charts Company. Retrieved August 25, 2014.
"ČNS IFPI" (in Slovak). Hitparáda – Radio Top 100 Oficiálna. IFPI Czech Republic. Note: insert 201445 into search. Retrieved September 19, 2014.
"ČNS IFPI" (in Slovak). Hitparáda – Singles Digital Top 100 Oficiálna. IFPI Czech Republic. Note: Select SINGLES DIGITAL - TOP 100 and insert 201444 into search. Retrieved September 26, 2014.
"SloTop50: Slovenian official singles weekly chart" (in Slovenian). SloTop50. Archived from the original on January 6, 2018. Retrieved January 31, 2018.
"EMA Top 10 Airplay: Week Ending 2014-10-07". Entertainment Monitoring Africa. Retrieved October 9, 2014.
"South Korea Gaon International Chart (Gaon Chart)". Gaon Chart. Archived from the original on October 17, 2014. Retrieved September 27, 2014.
"Taylor Swift – Shake It Off". Swiss Singles Chart. Retrieved October 5, 2014.
"Official Singles Chart Top 100". Official Charts Company. Retrieved August 25, 2014.
"Taylor Swift Chart History (Hot 100)". Billboard. Retrieved August 28, 2014.
"Taylor Swift Chart History (Adult Pop Songs)". Billboard. Retrieved November 13, 2014.
"Taylor Swift Chart History (Country Airplay)". Billboard. Retrieved August 28, 2014.
"Taylor Swift Chart History (Dance Club Songs)". Billboard. Retrieved January 4, 2015.
"Taylor Swift Chart History (Latin Airplay)". Billboard. Retrieved January 4, 2015.
"Taylor Swift Chart History (Pop Songs)". Billboard. Retrieved September 26, 2014.
"Taylor Swift Chart History (Rhythmic)". Billboard. Retrieved September 26, 2014.
"Taylor Swift Chart History (Global 200)". Billboard. Retrieved August 22, 2023.
"Taylor Swift – Shake It Off". AFP Top 100 Singles. Retrieved November 9, 2023.
"RIAS Top Charts Week 44 (27 Oct – 2 Nov 2023)". Recording Industry Association Singapore. Archived from the original on November 7, 2023. Retrieved November 7, 2023.
"End of Year Charts – ARIA Top 100 Singles 2014". Australian Recording Industry Association. Archived from the original on January 9, 2015. Retrieved February 1, 2015.
"Jahreshitparade Singles 2014" (in German). Hung Medien. Archived from the original on April 10, 2017. Retrieved January 6, 2020.
"2014 Year End Charts – Top Canadian Hot 100". Billboard. January 2, 2013. Archived from the original on March 4, 2016. Retrieved December 11, 2014.
"Top de l'année Top Singles 2014" (in French). Syndicat National de l'Édition Phonographique. Archived from the original on July 4, 2020. Retrieved August 12, 2020.
"Top 100 Single-Jahrescharts" (in German). GfK Entertainment. Archived from the original on March 3, 2016. Retrieved August 10, 2015.
"Single Top 100 – eladási darabszám alapján – 2014" (in Hungarian). Mahasz. Archived from the original on June 3, 2020. Retrieved March 18, 2020.
"IRMA – Best of Singles". Irish Recorded Music Association. Archived from the original on September 23, 2001. Retrieved December 30, 2014.
"סיכום 2014 בהשמעות רדיו: "מסתובב", אריק איינשטיין ופאר טסי". mako.co.il (in Hebrew). January 1, 2015. Archived from the original on March 3, 2018. Retrieved January 27, 2015.
"Japan Hot 100 – Year End 2014". Billboard. January 2, 2013. Archived from the original on November 27, 2015. Retrieved December 11, 2014.
"Top 100-Jaaroverzicht van 2014" (in Dutch). Dutch Top 40. Archived from the original on March 4, 2016. Retrieved January 1, 2015.
"Jaaroverzichten – Singles 2014" (in Dutch). Dutch Charts. Hung Medien. Archived from the original on January 12, 2015. Retrieved December 25, 2014.
"End of Year Charts 2014". Recorded Music New Zealand. Archived from the original on March 4, 2016. Retrieved December 9, 2015.
"Utwory, których słuchaliśmy w radiu – Airplay 2014" (in Polish). Polish Society of the Phonographic Industry. Archived from the original on December 2, 2015. Retrieved January 21, 2015.
"Top 100 Songs Annual 2014" (in Spanish). Productores de Música de España. Archived from the original on May 9, 2022. Retrieved May 9, 2022.
"Swiss Year-End Charts 2014". Hung Medien. Archived from the original on December 20, 2014. Retrieved January 3, 2015.
"End of Year Singles Chart Top 100 – 2014". Official Charts. Official Charts Company. Archived from the original on February 12, 2016. Retrieved December 9, 2015.
"Hot 100 Songs – Year End 2014". Billboard. January 2, 2013. Archived from the original on February 1, 2016. Retrieved December 11, 2014.
"Adult Contemporary Songs: Year End 2014". Billboard. January 2, 2013. Archived from the original on April 16, 2016. Retrieved December 9, 2014.
"Adult Pop Songs: Year End 2014". Billboard. January 2, 2013. Archived from the original on April 16, 2016. Retrieved December 9, 2014.
"Pop Songs: Year End 2014". Billboard. January 2, 2013. Archived from the original on April 16, 2016. Retrieved December 9, 2014.
"ARIA Charts – End of Year Charts – Top 100 Singles 2015". Australian Recording Industry Association. Archived from the original on January 24, 2016. Retrieved January 6, 2016.
"Rapports Annuels 2015" (in Dutch). Ultratop. Archived from the original on May 15, 2020. Retrieved June 16, 2020.
"Canadian Hot 100 Year End 2015". Billboard. Archived from the original on August 23, 2016. Retrieved December 9, 2015.
"Top de l'année Top Singles 2015" (in French). Syndicat National de l'Édition Phonographique. Archived from the original on July 3, 2020. Retrieved August 12, 2020.
"Single Top 100 – eladási darabszám alapján – 2015". Mahasz. Archived from the original on March 5, 2016. Retrieved March 18, 2020.
"Japan Hot 100 Year End 2015". Billboard. Archived from the original on December 14, 2015. Retrieved December 9, 2015.
"SloTop50: Slovenian official year end singles chart". slotop50.si. Archived from the original on March 3, 2016. Retrieved July 18, 2015.
"Hot 100: Year End 2015". Billboard. Archived from the original on August 23, 2016. Retrieved December 9, 2015.
"Adult Contemporary Songs Year End 2015". Billboard. Archived from the original on January 1, 2016. Retrieved December 9, 2015.
"Adult Pop Songs Year End 2015". Billboard. Archived from the original on December 14, 2015. Retrieved December 9, 2015.
"Japan Hot 100 : Year End 2016". Billboard. Archived from the original on November 20, 2017. Retrieved August 18, 2017.
"Japan Hot 100 : Year End 2017". Billboard. Archived from the original on March 30, 2018. Retrieved December 20, 2018.
"ARIA Top 100 Singles Chart for 2023". Australian Recording Industry Association. Archived from the original on January 12, 2024. Retrieved January 12, 2024.
"Billboard Global 200 – Year-End 2023". Billboard. Archived from the original on November 21, 2023. Retrieved November 21, 2023.
"2019 ARIA End of Decade Singles Chart". ARIA Charts. Archived from the original on January 11, 2020. Retrieved January 16, 2020.
Copsey, Rob (December 11, 2019). "The UK's Official Top 100 biggest songs of the decade". Official Charts Company. Archived from the original on December 11, 2019. Retrieved December 12, 2019.
"Decade-End Charts: Hot 100 Songs". Billboard. Archived from the original on November 14, 2019. Retrieved November 15, 2019.
"Digital Song Sales – Decade-End". Billboard. Archived from the original on August 17, 2021. Retrieved August 14, 2021.
"Streaming Songs – Decade-End". Billboard. Archived from the original on August 18, 2021. Retrieved August 14, 2021.
"Hot 100 turns 60". Billboard. Archived from the original on August 3, 2018. Retrieved August 6, 2018.
"Billboard Hot 100 60th Anniversary Interactive Chart". Billboard. Archived from the original on August 3, 2018. Retrieved December 10, 2018.
"Greatest of All Time Adult Pop Songs : Page 1". Billboard. Archived from the original on May 21, 2018. Retrieved March 16, 2018.
"Ultratop − Goud en Platina – singles 2018". Ultratop. Hung Medien. Retrieved January 10, 2021.
"Danish single certifications – Taylor Swift – Shake It Off". IFPI Danmark. Retrieved October 3, 2021.
"Gold-/Platin-Datenbank (Taylor Swift; 'Shake It Off')" (in German). Bundesverband Musikindustrie. Retrieved December 21, 2024.
"Certificaciones" (in Spanish). Asociación Mexicana de Productores de Fonogramas y Videogramas. Retrieved March 15, 2015. Type Taylor Swift in the box under the ARTISTA column heading and Shake It Off in the box under the TÍTULO column heading.
"New Zealand single certifications – Taylor Swift – Mirrorball". Radioscope. Retrieved December 19, 2024. Type Mirrorball in the "Search:" field.
"Portuguese single certifications – Taylor Swift – Shake It Off" (PDF) (in Portuguese). Associação Fonográfica Portuguesa. Retrieved September 21, 2020.
"Veckolista Singlar, vecka 45, 2014 | Sverigetopplistan" (in Swedish). Sverigetopplistan. Retrieved August 25, 2022. Scroll to position 12 to view certification.
"The Official Swiss Charts and Music Community: Awards ('Shake It Off')". IFPI Switzerland. Hung Medien. Retrieved November 19, 2018.
"Danish single certifications – Taylor Swift – Shake It Off". IFPI Danmark. Retrieved June 26, 2020.
"Japanese single streaming certifications – Taylor Swift – Shake It Off" (in Japanese). Recording Industry Association of Japan. Retrieved December 26, 2023. Select 2023年11月 on the drop-down menu
Melas, Chloe (November 16, 2020). "Taylor Swift Speaks Out about Sale of Her Masters". CNN. Archived from the original on November 18, 2020. Retrieved November 19, 2020.
"Taylor Swift Wants to Re-Record Her Old Hits". BBC News. August 22, 2019. Archived from the original on August 22, 2019. Retrieved August 22, 2019.
Finnis, Alex (November 17, 2020). "Taylor Swift Masters: The Controversy around Scooter Braun Selling the Rights to Her Old Music Explained". i. Archived from the original on February 12, 2021. Retrieved February 13, 2021.
Shah, Neil (April 9, 2021). "Taylor Swift Releases New Fearless Album, Reclaiming Her Back Catalog". The Wall Street Journal. Archived from the original on October 8, 2021. Retrieved September 25, 2022.
Vassell, Nicole (October 27, 2023). "Taylor Swift Fans Celebrate As Pop Star Releases 1989 (Taylor's Version)". The Independent. Archived from the original on October 30, 2023. Retrieved October 30, 2023.
D'Souza, Shaad (October 30, 2023). "Taylor Swift: 1989 (Taylor's Version) Album Review". Pitchfork. Archived from the original on October 30, 2023. Retrieved October 30, 2023.
Swift, Taylor (2023). 1989 (Taylor's Version) (Compact disc liner notes). Republic Records. 0245597656.
"ARIA Top 50 Singles Chart". Australian Recording Industry Association. November 6, 2023. Archived from the original on November 3, 2023. Retrieved November 3, 2023.
"Taylor Swift Chart History (Canadian Hot 100)". Billboard. Retrieved November 7, 2023.
"Taylor Swift Chart History (Global 200)". Billboard. Retrieved November 7, 2023.
"Digital Singles Chart (International)". IFPI Greece. Archived from the original on November 13, 2023. Retrieved November 8, 2023.
"NZ Top 40 Singles Chart". Recorded Music NZ. November 6, 2023. Archived from the original on November 3, 2023. Retrieved November 4, 2023.
"Taylor Swift Chart History (Philippines Songs)". Billboard. Archived from the original on November 8, 2023. Retrieved November 7, 2023.
"Veckolista Heatseeker, vecka 44". Sverigetopplistan. Archived from the original on November 3, 2023. Retrieved November 4, 2023.
"Official Singles Downloads Chart Top 100". Official Charts Company. Retrieved November 3, 2023.
"Official Singles Sales Chart Top 100". Official Charts Company. Archived from the original on November 11, 2023. Retrieved November 3, 2023.
"Official Streaming Chart Top 100". Official Charts Company. Archived from the original on November 3, 2023. Retrieved November 3, 2023.
"Taylor Swift Chart History (Hot 100)". Billboard. Retrieved November 7, 2023.
"Taylor Swift Chart History (Billboard Vietnam Hot 100)". Billboard. Archived from the original on June 6, 2022. Retrieved November 11, 2023.

    "Brazilian single certifications – Taylor Swift – Shake it Off (Taylor's Version)" (in Portuguese). Pro-Música Brasil. Retrieved July 24, 2024.

Bibliography

    Dubrofsky, Rachel (2016). "A Vernacular of Surveillance: Taylor Swift and Miley Cyrus Perform White Authenticity". Surveillance & Society. 14 (2): 184–196. doi:10.24908/ss.v14i2.6022.
    Fuller, Glen (2017). "The #tay4hottest100 new media event: discourse, publics and celebrity fandom as connective action". Communication Research and Practice. 4 (2): 167–182. doi:10.1080/22041451.2017.1295221. S2CID 157195033.
    McNutt, Myles (2020). "From 'Mine' to 'Ours': Gendered Hierarchies of Authorship and the Limits of Taylor Swift's Paratextual Feminism". Communication, Culture and Critique. 13 (1): 72–91. doi:10.1093/ccc/tcz042.
    Nobile, Drew (2015). "Counterpoint in Rock Music: Unpacking the 'Melodic-Harmonic Divorce'". Music Theory Spectrum. 37 (2): 189–203. doi:10.1093/mts/mtv019.
    Wilkinson, Maryn (2017). "Taylor Swift: the hardest working, zaniest girl in show business". Celebrity Studies. 10 (3): 441–444. doi:10.1080/19392397.2019.1630160.

    vte

Taylor Swift songs

    Singles discography

Taylor Swift	

    "Tim McGraw" "Picture to Burn" "Teardrops on My Guitar" "A Place in This World" "Should've Said No" "Our Song"

Fearless	

    "Fearless" "Fifteen" "Love Story" "Hey Stephen" "White Horse" "You Belong with Me" "Breathe" "You're Not Sorry" "The Way I Loved You" "Forever & Always" "The Best Day" "Change"

Taylor's Version	

    "You All Over Me" "Mr. Perfectly Fine" "That's When"

Speak Now	

    "Mine" "Sparks Fly" "Back to December" "Speak Now" "Dear John" "Mean" "The Story of Us" "Never Grow Up" "Enchanted" "Better than Revenge" "Innocent" "Haunted" "Long Live" "Ours"

Taylor's Version	

    "Electric Touch" "When Emma Falls in Love" "I Can See You" "Castles Crumbling"

Red	

    "State of Grace" "Red" "Treacherous" "I Knew You Were Trouble" "All Too Well" "22" "We Are Never Ever Getting Back Together" "The Last Time" "Holy Ground" "Everything Has Changed" "Begin Again"

Taylor's Version	

    "Nothing New" "Message in a Bottle" "I Bet You Think About Me" "Forever Winter"

1989	

    "Welcome to New York" "Blank Space" "Style" "Out of the Woods" "All You Had to Do Was Stay" "Shake It Off" "I Wish You Would" "Bad Blood" "Wildest Dreams" "How You Get the Girl" "This Love" "I Know Places" "Clean" "You Are in Love" "New Romantics"

Taylor's Version	

    "'Slut!'" "Say Don't Go" "Now That We Don't Talk" "Suburban Legends" "Is It Over Now?"

Reputation	

    "...Ready for It?" "End Game" "I Did Something Bad" "Don't Blame Me" "Delicate" "Look What You Made Me Do" "So It Goes..." "Gorgeous" "Getaway Car" "Dress" "Call It What You Want" "New Year's Day"

Lover	

    "I Forgot That You Existed" "Cruel Summer" "Lover" "The Man" "The Archer" "Miss Americana & the Heartbreak Prince" "Paper Rings" "Cornelia Street" "Death by a Thousand Cuts" "London Boy" "Soon You'll Get Better" "False God" "You Need to Calm Down" "Me!"

Folklore	

    "The 1" "Cardigan" "The Last Great American Dynasty" "Exile" "My Tears Ricochet" "Mirrorball" "Seven" "August" "This Is Me Trying" "Illicit Affairs" "Invisible String" "Mad Woman" "Epiphany" "Betty" "Peace" "Hoax" "The Lakes"

Evermore	

    "Willow" "Champagne Problems" "'Tis the Damn Season" "Tolerate It" "No Body, No Crime" "Happiness" "Dorothea" "Coney Island" "Long Story Short" "Marjorie"

Midnights	

    "Lavender Haze" "Maroon" "Anti-Hero" "Snow on the Beach" "You're on Your Own, Kid" "Midnight Rain" "Question...?" "Vigilante Shit" "Bejeweled" "Labyrinth" "Karma" "Sweet Nothing" "Mastermind" "Hits Different" "Bigger Than the Whole Sky" "Would've, Could've, Should've" "You're Losing Me"

The Tortured Poets
Department	

    "Fortnight" "The Tortured Poets Department "My Boy Only Breaks His Favorite Toys" "Down Bad" "So Long, London" "But Daddy I Love Him" "Fresh Out the Slammer" "Florida!!!" "Guilty as Sin?" "Who's Afraid of Little Old Me?" "Loml" "I Can Do It with a Broken Heart" "The Smallest Man Who Ever Lived" "The Alchemy" "Clara Bow" "The Black Dog" "So High School" "Thank You Aimee"

Soundtrack songs	

    "Crazier" "Today Was a Fairytale" "Safe & Sound" "Eyes Open" "Sweeter than Fiction" "I Don't Wanna Live Forever" "Beautiful Ghosts" "Only the Young" "Carolina"

Featured songs	

    "Two Is Better Than One" "Half of My Heart" "Both of Us" "Babe" "Gasoline" "Renegade" "The Joker and the Queen" "The Alcott" "Us"

Other songs	

    "Best Days of Your Life" "Ronan" "Highway Don't Care" "Better Man" "Christmas Tree Farm" "All of the Girls You Loved Before"

    Category

Authority control databases Edit this at Wikidata	

    MusicBrainz release group

Categories:

    2014 singles2014 songsBig Machine Records singlesBillboard Hot 100 number-one singlesCanadian Hot 100 number-one singlesAmerican dance-pop songsMusic videos directed by Mark RomanekNumber-one singles in AustraliaNumber-one singles in HungaryNumber-one singles in New ZealandSong recordings produced by Max MartinSong recordings produced by Shellback (record producer)Song recordings produced by Taylor SwiftSong recordings produced by Chris RoweSongs written by Taylor SwiftSongs written by Max MartinSongs written by Shellback (record producer)Songs involved in plagiarism controversiesTaylor Swift songsNumber-one singles in PolandReese Witherspoon songsRyan Adams songs

    This page was last edited on 30 December 2024, at 13:53 (UTC).
    Text is available under the Creative Commons Attribution-ShareAlike 4.0 License; additional terms may apply. By using this site, you agree to the Terms of Use and Privacy Policy. Wikipedia® is a registered trademark of the Wikimedia Foundation, Inc., a non-profit organization.

    Privacy policy
    About Wikipedia
    Disclaimers
    Contact Wikipedia
    Code of Conduct
    Developers
    Statistics
    Cookie statement
    Mobile view

    Wikimedia Foundation
    Powered by MediaWiki



Wikipedia The Free Encyclopedia

    Donate
    Create account
    Log in

Contents
(Top)
Background

Lyrics and music

Release and commercial performance

Critical reception

    Accolades

Music video

    Development and release
    Reception

Live performances and other versions

Credits and personnel

Charts

    Weekly charts
    Year-end charts
    Decade-end charts
    All-time charts

Certifications

Release history

"Blank Space (Taylor's Version)"

    Reception
    Personnel
    Charts
    Certification

See also

References

        Sources

Blank Space

    Article
    Talk

    Read
    Edit
    View history

Tools

Appearance
Text

    Small
    Standard
    Large

Width

    Standard
    Wide

Color (beta)

    Automatic
    Light
    Dark

Featured article
From Wikipedia, the free encyclopedia
This article is about the Taylor Swift song. For other uses, see Blank space (disambiguation).
"Blank Space"
Cover artwork of "Blank Space", a polaroid photo of Swift leaning on a bench
Single by Taylor Swift
from the album 1989
Released	November 10, 2014
Studio	

    MXM (Stockholm, Sweden)
    Conway (Los Angeles, US)

Genre	Electropop
Length	3:52
Label	Big Machine
Songwriter(s)	

    Taylor Swift Max Martin Shellback

Producer(s)	

    Max Martin Shellback

Taylor Swift singles chronology
"Shake It Off"
(2014) 	"Blank Space"
(2014) 	"Style"
(2015)
Music video
"Blank Space" on YouTube

"Blank Space" is a song by the American singer-songwriter Taylor Swift and the second single from her fifth studio album, 1989 (2014). Swift wrote the song with its producers, Max Martin and Shellback. Inspired by the media scrutiny on Swift's love life that affected her girl-next-door reputation, "Blank Space" portrays a flirtatious woman with multiple romantic attachments. It is an electropop track with a minimal arrangement consisting of synthesizers, hip hop–influenced beats, and layered vocals.

Big Machine in partnership with Republic Records released "Blank Space" to US radio on November 10, 2014. One of the best-selling singles of 2015, it topped charts in Australia, Canada, Iceland, Scotland, and South Africa. In the United States, it spent seven weeks atop the Billboard Hot 100 and was certified eight times platinum by the Recording Industry Association of America (RIAA). Music critics praised the production and Swift's songwriting; some picked it as 1989's highlight. The song earned three nominations at the 58th Grammy Awards, including two general categories: Record of the Year and Song of the Year. Rolling Stone placed it at number 320 on their 2024 revision of the 500 Greatest Songs of All Time.

Joseph Kahn directed the music video for "Blank Space", which depicts Swift as a jealous woman who acts erratically when she suspects her boyfriend's infidelity. The video won Best Pop Video and Best Female Video at the 2015 MTV Video Music Awards, and it ranked 67th on Rolling Stone's 100 Greatest Music Videos of All Time. Swift included "Blank Space" in the set lists for three of her world tours: the 1989 World Tour (2015), Reputation Stadium Tour (2018), and the Eras Tour (2023–2024). The song was covered by several rock musicians. Following the 2019 dispute regarding the ownership of Swift's back catalog, she re-recorded the song as "Blank Space (Taylor's Version)" for her 2023 re-recorded album 1989 (Taylor's Version).
Background

Inspired by 1980s synth-pop with synthesizers, drum pads, and overlapped vocals, Taylor Swift abandoned the country stylings of her previous releases to incorporate a pop production for her fifth studio album, 1989, which was released in 2014.[1][2][3] Swift began writing songs for the album in mid-2013 concurrently with the start of Swift's headlining world tour in support of her fourth studio album Red.[4] On 1989, Swift and the Swedish producer Max Martin served as executive producers.[2] Martin and his frequent collaborator Shellback produced seven out of 13 songs on the album's standard edition.[5]

Having been known as "America's Sweetheart" thanks to her wholesome and down-to-earth girl next door image,[6][7] Swift saw her reputation blemished due to her history of romantic relationships with a series of high-profile celebrities. The New York Times asserted in 2013 that her "dating history [had] begun to stir what feels like the beginning of a backlash", questioning whether Swift was in the midst of a quarter-life crisis.[8] The Tampa Bay Times observed that until the release of 1989, Swift's love life had become a fixed tabloid interest and overshadowed her musicianship.[7] Swift disliked the media portrayal of her as a "serial-dater", feeling that it undermined her professional works, and became reticent to discuss her personal life in public.[9][10] The tabloid scrutiny on her image prompted her to write satirical songs about her perceived image, in addition to her traditional romantic themes.[11]
Lyrics and music
"Blank Space"
Duration: 22 seconds.0:22
An electropop song set over minimal hip hop-influenced beats, "Blank Space" satirizes Swift's image as a seductive woman with a long history of relationships.
Problems playing this file? See media help.

Talking to GQ in 2015, Swift said that she envisioned "Blank Space" to be a satirical self-referential nod to the media perception of her image as "a girl who's crazy but seductive but glamorous but nuts but manipulative".[12] She admitted that she had felt personally attacked for a long time before realizing "it was kind of hilarious".[12] She co-wrote the song with its producers, Max Martin and Shellback.[5]

"Blank Space" follows the verse–chorus song structure.[13] The lyrics in the verses are clipped, "Magic, madness, heaven, sin", which the musicologist Nate Sloan said to set a mysterious and dreadful tone.[13] At one point, Swift describes herself as a "nightmare dressed like a daydream".[14] The refrain alludes to Swift's songwriting practice taking inspiration from her love life: the lyrics, "Got a long list of ex-lovers They'll tell you I'm insane But I've got a blank space, baby", are followed by a brief silence and then a clicking retractable pen sound, and Swift concludes the refrain: "And I'll write your name."[15] After the song's release, the line "Got a long list of ex-lovers" was misheard by some audience as "All the lonely Starbucks lovers", which prompted internet discussions including a response from Starbucks themselves.[16][17]

Swift told NME in 2015 that when "Blank Space" was released, "[half] the people got the joke, half the people really think that [she]was like really owning the fact that [she was] a psychopath".[18] According to Sloan, the narrator of "Blank Space" is unreliable, and therefore it is open to interpretation whether the song is a true portrayal of Swift's character or not.[19] In contemporary publications, journalists commented that the track represented 1989's lighthearted view on failed relationships and departed from the idealized romance on Swift's past albums.[20][21][22] Others wrote that Swift made fun of her image and the media discourse surrounding her celebrity, which later served as the foundation for her sixth studio album Reputation (2017), an album exploring her public experiences and the media gossip.[23][24]

Martin and Shellback employed a sparse production for "Blank Space" as Swift wanted the song to emphasize the lyrics and vocals.[2] Musically, "Blank Space" is an electropop song[25][26] that is set over minimal hip hop–influenced beats.[27] Annie Zaleski said that the beats resonate like the sounds of a grandfather clock.[26] The song incorporates synthesizers, percussioned guitar strums, and layered backing vocals.[21][28] Swift sing-speaks the verses[26] and, in the refrain, sings in her higher register as the production crescendos with faster programmed drums.[13] Some critics compared the song's minimal production to the music of Lorde, specifically her 2013 album Pure Heroine.[14][21][29] According to Spin's Andrew Unterberger, "Blank Space" embraces 1980s pop music authenticity but also with a modern twist.[29]
Release and commercial performance

"Blank Space" was the second single from 1989. In the United States, Big Machine and Republic Records released the song to rhythmic crossover radio on November 10,[30] and hot adult contemporary[31] and contemporary hit radio on November 11, 2014.[32] Universal sent "Blank Space" to Italian radio on December 12, 2014,[33] and released a CD single version of "Blank Space" in Germany on January 2, 2015.[34]

"Blank Space" debuted at number 18 on the Billboard Hot 100 chart dated November 15, 2014.[35] The single reached number one in its third week on the chart, supported by the release of its music video. It took the number-one position from 1989's lead single "Shake It Off", making Swift the first woman to succeed herself at the top spot.[36] "Blank Space" remained atop the Billboard Hot 100 for seven consecutive weeks.[37] In August 2023, "Blank Space" re-entered the Hot 100 and reached number 46 after it increased in streams; this was brought by Swift's announcement of the re-recorded album 1989 (Taylor's Version) and her performances of the song on her Eras Tour.[38][39] The Recording Industry Association of America (RIAA) certified "Blank Space" eight-times platinum, which denotes eight million units based on sales and track-equivalent on-demand streams, in July 2018,[40] and the single had sold 4.6 million digital copies in the United States by October 2022.[41]

The single also reached number one in Australia,[42] Canada,[43] South Africa,[44] and Scotland.[45] It peaked atop Billboard's Euro Digital Song Sales[46] and the Finnish Download Chart.[47] "Blank Space" charted within the top five of national record charts, at number two in New Zealand,[48] Poland,[49] Slovakia,[50] number three in Bulgaria,[51] number four in the Czech Republic,[52] Ireland,[53] Israel,[54] the United Kingdom,[55] and number five in Lebanon.[56] The track received multi-platinum certifications in many countries, including fourteen-times platinum in Australia[57] and four-times diamond in Brazil.[58] It was certified four-times platinum in Canada,[59] Poland,[60] and New Zealand;[61] triple platinum in the United Kingdom;[62] and double platinum in Austria[63] and Portugal.[64] According to the International Federation of the Phonographic Industry (IFPI), "Blank Space" was the eighth-best-selling song of 2015, selling 9.2 million units.[65]
Critical reception

The song received widespread acclaim. Upon the release of 1989, Shane Kimberline of musicOMH called "Blank Space" one of the album's best songs.[66] PopMatters's Corey Baesley lauded it as "easily a candidate for the best pop song of 2014", writing that the minimal production may "sound bright and easy" but was in fact "weapons-grade, professional pop".[21] Sydney Gore from The 405 deemed "Blank Space" the album's highlight,[14] and Aimee Cliff from Fact labeled it one of Swift's "most enjoyable songs to date" for portraying Swift's love life in a larger-than-life manner.[67] Drowned in Sound's Robert Leedham wrote that Swift succeeded in experimenting with new musical styles on 1989, specifically choosing "Blank Space" as an example.[68]

The Observer critic Kitty Empire picked "Blank Space" as a song that showcased Swift's musical and lyrical maturity, calling it "an out-and-out pop song with an intriguingly skeletal undercarriage".[69] Writing for the Los Angeles Times, Mikael Wood selected the track as one of the album's better songs because of Swift's songwriting craftsmanship.[27] The New York Times critic Jon Caramanica deemed the song "Swift at her peak" that "serves to assert both her power and her primness".[70] The Independent's Andy Gill was less enthusiastic, calling it a "corporate rebel clichéd [sic]" song.[25]

Retrospective reviews of "Blank Space" have been positive. Alexis Petridis of The Guardian in 2019 declared "Blank Space" the best single Swift had released, praising its success in transforming Swift's image from a country singer-songwriter to a pop star thanks to its "effortless" melody and witty lyrics.[28] Rolling Stone reviewer Rob Sheffield wrote: "Every second of 'Blank Space' is perfect."[71] Paste in 2020 described the song as "remarkably well-made, infectiously catchy, and legitimately funny", and named it the best song on 1989.[72] Selja Rankin from Entertainment Weekly also dubbed "Blank Space" the best track on the album, praising the over-the-top lyrics and its catchy 1980s pop sound.[73] The Recording Academy in 2023 picked "Blank Space" as one of Swift's 13 essential songs that represented her songwriting and musicianship.[74]
Accolades

Rolling Stone ranked "Blank Space" sixth on their list of the best songs of 2014,[75] 73rd on their list of the best songs of the 2010s decade,[76] 357th on their 2021 revision of the 500 Greatest Songs of All Time,[77] and later at 320 in their 2024 revised list.[78] Time named it as the ninth best song in their year-end list.[79] The song placed at number three on The Village Voice's annual year-end Pazz & Jop critics' poll of 2014.[80] Stereogum[81] and Uproxx[82] ranked the song at numbers 49 and 72 on their lists of the best songs of the 2010s decade, respectively. Billboard named it one of the 100 "Songs That Defined the Decade". Katie Atkinson wrote that the single consolidated Swift's trademark autobiographical storytelling in music while "setting the standard for a new, self-aware pop star" in poking fun at her perceived image.[83] On Slant Magazine's list of the 100 best singles of the 2010s, "Blank Space" ranked 15th.[84]

"Blank Space" won Song of the Year at the 2015 American Music Awards.[85] At the 2016 BMI Awards, the song was one of the Award-Winning Songs that helped Swift earn the honor Songwriter of the Year.[86] It earned a nomination for International Work of the Year at the 2016 APRA Awards in Australia.[87] At the 58th Annual Grammy Awards in 2016, "Blank Space" was nominated in three categories—Record of the Year, Song of the Year, and Best Pop Solo Performance.[88]
Music video
Development and release
Exterior view of Oheka Castle and its gardens
The video was primarily shot at Oheka Castle in Long Island, New York.

Joseph Kahn directed the music video for "Blank Space". According to Kahn, Swift conceptualized the video to "[address] this concept of, if she has so many boys breaking up with her maybe the problem isn't the boy, maybe the problem is her".[89] Photography took place at two locations on Long Island: primary shooting took place at Oheka Castle, with a few additional scenes shot at Woolworth Estate. The video was shot over three days in September 2014.[90] The last day was dedicated to film American Express Unstaged: Taylor Swift Experience, an interactive 360° mobile app in collaboration with American Express.[91] Kahn told Mashable that Swift was thorough in choosing the visual devices and imagery: "When you have an artist wanting to test her imaging, it's always great territory to be in."[89]

Kahn took inspirations from Stanley Kubrick's 1971 film A Clockwork Orange for the video's symmetrical framing style.[91] The video begins as the male love interest (Sean O'Pry) drives an AC Cobra towards the mansion of Swift's character. They quickly become a loving couple: they dance together, paint a portrait for the boyfriend, walk along the estate grounds, and ride horses.[92] Halfway through the video, Swift's character notices him texting someone, and the couple begins to fall apart: they begin to fight and Swift's character shows erratic behaviors such as throwing vases, slashing the painted portrait, and burning her boyfriend's clothes, which drives him to end the relationship.[92] Before the boyfriend leaves the mansion, Swift's character smashes her boyfriend's car using a golf club, a reference to Tiger Woods's 2009 cheating scandal.[89] After he drives away, a new man (Andrea Denver) approaches, offering Swift a new hope for love.[92]

Swift planned to premiere the video on Good Morning America on November 11, 2014, but Yahoo! accidentally leaked it a day before; Swift posted the video onto her Vevo account quickly afterwards.[93] The interactive app American Express Unstaged: Taylor Swift Experience, featuring the 360° video version of "Blank Space", was released for free onto mobile app stores. The user can choose to either follow Swift and her love interest throughout the linear storyline, or leave the storyline to explore other rooms in the mansion and find interactive easter eggs, such as Swift's childhood photos.[94][95] Kahn told Rolling Stone that the app was created with "superfans" who wanted to "feel even closer to Swift" in mind.[90]
Reception

Some media outlets compared the narrative of "Blank Space" to that of Gone Girl, citing that both Swift's character and Gone Girl's protagonist "[strip] away the romantic sheen she's given all her relationships in the past".[96] Randall Roberts from the Los Angeles Times wrote that Swift delivered an "Oscar-worthy" performance.[92] Billboard praised the video's cinematic quality and aesthetics and found Swift's self-referential portrayal amusing, which served as "icing on the blood-filled cake".[97] The Guardian's columnist Jessica Valenti complimented Swift's portrayal of her perceived image and dubbed the video "a feminist daydream", where "the narrow and sexist caricatures attached to women are acted out for our amusement, their full ridiculousness on display".[98]

USA Today and Spin in 2017 deemed "Blank Space" the greatest video Swift had done;[99] the latter praised the aesthetics as glamorous and lauded the hilarious depiction of Swift's reputation.[100] Entertainment Weekly in 2020 picked "Blank Space" as the best video among the 1989 singles, describing it as "the only music video that can be earnestly described as 'Kubrickian'".[101] It won Best Pop Video and Best Female Video at the 2015 MTV Video Music Awards[102] and earned a nomination for Best International Female Video at the MTV Video Music Awards Japan.[103] The American Express Unstaged: Taylor Swift Experience app won Original Interactive Program at the 67th Primetime Creative Arts Emmy Awards.[104] Rolling Stone placed "Blank Space" at number 67 on its list of the 100 Greatest Music Videos of All Time.[105]
Live performances and other versions
Taylor Swift on the 1989 World Tour
Swift performing "Blank Space" during the 1989 World Tour

Swift performed "Blank Space" during the "1989 Secret Session", live streamed by Yahoo! and iHeartRadio on October 27, 2014.[106] Swift premiered the song on television at the 2014 American Music Awards, where she recreated the narrative of the music video, acting as a psychopathic woman who acts erratically towards her boyfriend.[107] She again performed the song on The Voice on November 25,[108] at the 2014 Victoria's Secret Fashion Show on December 2,[109] and during Capital FM's Jingle Bell Ball 2014 in London, broadcast on December 5.[110]

On February 25, 2015, Swift opened the 2015 Brit Awards with a rendition of "Blank Space". At the beginning of the performance, Swift sang the song in front of a white background featuring silhouettes of backup dancers.[111] The song was part of the set lists for three of Swift's concert tours—the 1989 World Tour (2015),[112] Reputation Stadium Tour (2018),[113] and the Eras Tour (2023–2024).[114] On September 9, 2019, Swift performed the song at the City of Lover one-off concert in Paris, France.[115] She performed the song again at the We Can Survive charity concert on October 19, 2019, in Los Angeles.[116] At the 2019 American Music Awards, where Swift was honored Artist of the Decade, she performed "Blank Space" as part of a medley of her hits.[117] She again performed the song at Capital FM's Jingle Bell Ball 2019 in London,[118] and at iHeartRadio Z100's Jingle Ball 2019 in New York City.[119]

Following the song's debut at the 2014 American Music Awards, the rapper Pitbull uploaded a remix featuring his rap verse to SoundCloud on December 15, 2014.[120] The retro music group Postmodern Jukebox transformed the song into a 1940s-inspired track in their cover,[121] and the rock band Imagine Dragons performed a slowed down rendition of the song sampling Ben E. King's "Stand by Me" at BBC Radio 1 Live Lounge in February 2015.[122] I Prevail, another rock band, released a post-hardcore cover of "Blank Space" as their debut single in December 2014.[123] The cover reached number nine on Billboard Hot Rock Songs[124] and number 90 on the Billboard Hot 100,[125] and received a platinum certification by the RIAA, which denotes one million track-equivalent units.[126] It was also certified Gold by the Australian Recording Industry Association (ARIA) in 2024.[57]

The rock singer Ryan Adams covered "Blank Space" on his 2015 track-by-track cover album of Swift's 1989.[127] On his rendition, Adams incorporated stripped-down, acoustic string instruments, contrasting the original's electronic production.[128][129] The indie singer Father John Misty released a cover version of the song in the style of the rock band the Velvet Underground in 2015.[130] The cover is a reinterpretation of Adams's version and is built on the melody of the song "I'm Waiting for the Man".[131]
Credits and personnel

Credits adapted from the liner notes of 1989[5]

    Taylor Swift – vocals, background vocals, songwriter, shouts
    Cory Bice – recording assistant
    Tom Coyne – mastering
    Serban Ghenea – mixing
    John Hanes – mixing engineer
    Sam Holland – recording
    Michael Ilbert – audio recording
    Max Martin – producer, songwriter, keyboards, programming
    Shellback – producer, songwriter, acoustic guitar, electric guitar, bass, keyboards, percussion, programming, shouts, stomps

Charts
Weekly charts
2014–2015 weekly chart performance for "Blank Space" Chart (2014–2015) 	Peak
position
Australia (ARIA)[42] 	1
Austria (Ö3 Austria Top 40)[132] 	6
Belgium (Ultratop 50 Flanders)[133] 	13
Belgium (Ultratop 50 Wallonia)[134] 	26
Brazil (Billboard Hot 100)[135] 	59
Bulgaria (IFPI)[51] 	3
Canada (Canadian Hot 100)[43] 	1
Canada AC (Billboard)[136] 	2
Canada CHR/Top 40 (Billboard)[137] 	1
Canada Hot AC (Billboard)[138] 	1
CIS Airplay (TopHit)[139] 	84
Czech Republic (Rádio – Top 100)[52] 	4
Euro Digital Song Sales (Billboard)[46] 	1
Finland Airplay (Radiosoittolista)[140] 	3
Finland Download (Latauslista)[47] 	1
France (SNEP)[141] 	27
France Airplay (SNEP)[142] 	18
Germany (GfK)[143] 	9
Greece Digital Songs (Billboard)[144] 	2
Hungary (Single Top 40)[145] 	7
Iceland (RÚV)[146] 	1
Ireland (IRMA)[53] 	4
Israel (Media Forest)[54] 	4
Italy (FIMI)[147] 	27
Italy Digital Song Sales (Billboard)[148] 	10
Japan (Japan Hot 100)[149] 	45
Japan Adult Contemporary (Billboard)[150] 	10
Lebanon (Lebanese Top 20)[56] 	5
Luxembourg Digital Song Sales (Billboard)[151] 	4
Mexico (Billboard Mexican Airplay)[152] 	2
Mexico Anglo (Monitor Latino)[153] 	2
Netherlands (Dutch Top 40)[154] 	17
Netherlands (Single Tip)[155] 	3
New Zealand (Recorded Music NZ)[48] 	2
Poland (Polish Airplay Top 100)[49] 	2
Portugal Digital Song Sales (Billboard)[156] 	5
Romania (Airplay 100)[157] 	34
Scotland (OCC)[45] 	1
Slovakia (Rádio Top 100)[50] 	2
Slovenia (SloTop50)[158] 	9
South Africa (EMA)[44] 	1
South Korea International Singles (Gaon)[159] 	54
Spain (PROMUSICAE)[160] 	16
Switzerland (Schweizer Hitparade)[161] 	12
Ukraine Airplay (TopHit)[162] 	66
UK Singles (OCC)[55] 	4
US Billboard Hot 100[163] 	1
US Adult Contemporary (Billboard)[164] 	1
US Adult Pop Airplay (Billboard)[165] 	1
US Dance Club Songs (Billboard)[166] 	23
US Latin Airplay (Billboard)[167] 	48
US Pop Airplay (Billboard)[168] 	1
US Rhythmic (Billboard)[169] 	14
Venezuela (Record Report)[170] 	2
2021–2024 weekly chart performance for "Blank Space" Chart (2021–2024) 	Peak
position
Germany (GfK Entertainment Charts)[143] 	23
Global 200 (Billboard)[171] 	32
Greece International (IFPI Greece)[172] 	64
Italy (FIMI)[173] 	89
Malaysia International (RIM)[174] 	15
Netherlands (Single Tip)[175] 	17
New Zealand Catalogue Singles (RMNZ)[176] 	13
Philippines (Billboard)[177] 	25
Poland (Polish Airplay Top 100)[178] 	56
Portugal (AFP)[179] 	18
Singapore (RIAS)[180] 	5
Sweden Heatseeker (Sverigetopplistan)[181] 	7
Switzerland (Schweizer Hitparade)[161] 	20
UK Audio Streaming (OCC)[182] 	43
US Billboard Hot 100[183] 	46
Vietnam (Vietnam Hot 100)[184] 	57
	
Year-end charts
2014 year-end charts for "Blank Space" Chart (2014) 	Position
Australia (ARIA)[185] 	28
Hungary (Single Top 40)[186] 	92
New Zealand (Recorded Music NZ)[187] 	46
UK Singles (OCC)[188] 	64
2015 year-end charts for "Blank Space" Chart (2015) 	Position
Australia (ARIA)[189] 	55
Austria (Ö3 Austria Top 40)[190] 	58
Belgium (Ultratop Flanders)[191] 	73
Belgium (Ultratop Wallonia)[192] 	84
Canada (Canadian Hot 100)[193] 	6
France (SNEP)[194] 	173
Germany (Official German Charts)[195] 	80
Hungary (Single Top 40)[196] 	58
Netherlands (Dutch Top 40)[197] 	83
Netherlands (NPO 3FM)[198] 	76
Poland (ZPAV)[199] 	33
Slovenia (SloTop50)[200] 	15
Switzerland (Schweizer Hitparade)[201] 	67
UK Singles (Official Charts Company)[202] 	67
US Billboard Hot 100[203] 	7
US Adult Contemporary (Billboard)[204] 	5
US Adult Top 40 (Billboard)[205] 	9
US Mainstream Top 40 (Billboard)[206] 	6
2016 year-end chart for "Blank Space" Chart (2016) 	Position
Brazil (Brasil Hot 100)[207] 	74
2023 year-end charts for "Blank Space" Chart (2023) 	Position
Australia (ARIA)[208] 	59
Global 200 (Billboard)[209] 	67
Decade-end charts
2010s decade-end charts for "Blank Space" Chart (2010–2019) 	Position
Australia (ARIA)[210] 	84
US Billboard Hot 100[211] 	65
All-time charts
All-time charts for "Blank Space" Chart 	Position
US Billboard Hot 100[212] 	387
US Mainstream Top 40 (Billboard)[213] 	90

Certifications
Certifications for "Blank Space" Region 	Certification 	Certified units/sales
Australia (ARIA)[57] 	14× Platinum 	980,000‡
Austria (IFPI Austria)[63] 	2× Platinum 	60,000*
Brazil (Pro-Música Brasil)[58] 	4× Diamond 	1,000,000‡
Canada (Music Canada)[59] 	4× Platinum 	320,000*
Denmark (IFPI Danmark)[214] 	Platinum 	90,000‡
Germany (BVMI)[215] 	Platinum 	600,000‡
Italy (FIMI)[216] 	Platinum 	50,000‡
Japan (RIAJ)[217] 	Gold 	100,000*
Mexico (AMPROFON)[218] 	Gold 	30,000*
New Zealand (RMNZ)[61] 	5× Platinum 	150,000‡
Norway (IFPI Norway)[219] 	Platinum 	60,000‡
Poland (ZPAV)[60] 	4× Platinum 	200,000‡
Portugal (AFP)[64] 	2× Platinum 	40,000‡
Spain (PROMUSICAE)[220] 	Platinum 	60,000‡
Switzerland (IFPI Switzerland)[221] 	Gold 	15,000‡
United Kingdom (BPI)[62] 	3× Platinum 	1,800,000‡
United States (RIAA)[40] 	8× Platinum 	8,000,000‡
Streaming
Greece (IFPI Greece)[172] 	Platinum 	2,000,000†
Japan (RIAJ)[222] 	Gold 	50,000,000†

* Sales figures based on certification alone.
‡ Sales+streaming figures based on certification alone.
† Streaming-only figures based on certification alone.
Release history
Release dates and formats for "Blank Space" Region 	Date 	Format 	Label(s) 	Ref.
United States 	November 10, 2014 	Rhythmic radio 	

    Big MachineRepublic

	[30]
November 11, 2014 	Contemporary hit radio 	[32]
Hot adult contemporary radio 	Republic 	[31]
Italy 	December 12, 2014 	Radio airplay 	Universal 	[33]
Germany 	January 2, 2015 	CD single 	[34]
"Blank Space (Taylor's Version)"
"Blank Space (Taylor's Version)"
Song by Taylor Swift
from the album 1989 (Taylor's Version)
Released	October 27, 2023
Studio	Prime Recording (Nashville)
Length	3:51
Label	Republic
Songwriter(s)	

    Taylor Swift Max Martin Shellback

Producer(s)	

    Taylor Swift Christopher Rowe

Lyric video
"Blank Space (Taylor's Version)" on YouTube

After signing a new contract with Republic Records in 2018, Swift began re-recording her first six studio albums in November 2020.[223] The decision followed a public 2019 dispute between Swift and the music executive Scooter Braun, who acquired Big Machine Records, including the masters of Swift's albums which the label had released.[224][225] By re-recording the albums, Swift had full ownership of the new masters, which enabled her to control the licensing of her songs for commercial use and therefore substituted the Big Machine–owned masters.[226]

The re-recording of "Blank Space", subtitled "Taylor's Version", was released as part of 1989's re-recording, 1989 (Taylor's Version), on October 27, 2023.[227] Swift produced "Blank Space (Taylor's Version)" with Christopher Rowe, who had produced her previous re-recordings.[228] The track was engineered by Derek Garten at Prime Recording Studio in Nashville, Tennessee; mixed by Ghenea at MixStar Studios in Virginia Beach, Virginia; and mastered by Randy Merrill at Sterling Sound in Edgewater, New Jersey. Rowe and Sam Holland recorded Swift's vocals at Conway Recording Studios in Los Angeles and Kitty Committee Studio in New York.[229]
Reception

While giving positive reviews, music critics had different opinions on the re-recording's production. The Line of Best Fit journalist Kelsey Barnes commented that "Blank Space (Taylor's Version)" sounded like an "exact replica",[230] but The Independent's Adam White wrote that the re-recorded song features Swift's matured vocals that eroded the "raw mania" of the original song.[231] In NME, Hollie Geraghty praised the re-recording for showcasing one of the album's "deliciously polished belters that still feel brand new nearly a decade later".[232] "Blank Space (Taylor's Version)" peaked at number nine on the Billboard Global 200 chart.[233] On national singles charts, the re-recorded song peaked within the top 20 in Australia (9),[234] Canada (11),[43] New Zealand (12),[235] and the United States (12).[236]
Personnel

Credits adapted from the liner notes of 1989 (Taylor's Version)[229]

    Taylor Swift – lead vocals, background vocals, songwriter, producer
    Matt Billingslea – drums programming, membranophone, electric guitar, synthesizer
    Bryce Bordone – engineer for mix
    Dan Burns – synth bass programming, synth programming, additional engineer
    Derek Garten – additional programming, engineer, editor
    Serban Ghenea – mixing
    Sam Holland – vocals recording
    Max Martin – songwriter
    Mike Meadows – acoustic guitar, electric guitar, synthesizer
    Randy Merrill – mastering
    Christopher Rowe – producer, background vocals, vocals recording
    Shellback – songwriter

Charts
Chart performance for "Blank Space (Taylor's Version)" Chart (2023) 	Peak
position
Australia (ARIA)[234] 	9
Brazil (Brasil Hot 100)[237] 	63
Canada (Canadian Hot 100)[43] 	11
France (SNEP)[238] 	165
Global 200 (Billboard)[171] 	9
Greece International (IFPI)[239] 	16
Ireland (Billboard)[240] 	12
Malaysia (Billboard)[241] 	25
Malaysia International (RIM)[242] 	4
MENA (IFPI)[243] 	16
New Zealand (Recorded Music NZ)[235] 	12
Philippines (Billboard)[177] 	7
Spain (PROMUSICAE)[244] 	100
Sweden (Sverigetopplistan)[245] 	66
UAE (IFPI)[246] 	7
UK (Billboard)[247] 	15
UK Singles Downloads (OCC)[248] 	35
UK Singles Sales (OCC)[249] 	40
UK Streaming (OCC)[250] 	17
US Billboard Hot 100[163] 	12
Vietnam (Vietnam Hot 100)[251] 	49
Certification
Certification for "Blank Space (Taylor's Version)" Region 	Certification 	Certified units/sales
Australia (ARIA)[57] 	Gold 	35,000‡
Brazil (Pro-Música Brasil)[252] 	Gold 	20,000‡
New Zealand (RMNZ)[253] 	Gold 	15,000‡
United Kingdom (BPI)[254] 	Silver 	200,000‡

‡ Sales+streaming figures based on certification alone.
See also

    List of Billboard Hot 100 number ones of 2014
    List of Billboard Hot 100 number ones of 2015
    List of Billboard Adult Contemporary number ones of 2015
    List of Canadian Hot 100 number-one singles of 2014
    List of highest-certified singles in Australia
    List of number-one singles of 2014 (Australia)
    List of number-one singles of 2015 (South Africa)
    List of most-viewed YouTube videos

References

Eells, Josh (September 16, 2014). "Taylor Swift Reveals Five Things to Expect on 1989". Rolling Stone. Archived from the original on November 16, 2018. Retrieved November 16, 2018.
Zollo, Paul (February 13, 2016). "The Oral History of Taylor Swift's 1989". The Recording Academy. Archived from the original on April 4, 2016. Retrieved March 23, 2016 – via Cuepoint.
Light, Alan (December 5, 2014). "Billboard Woman of the Year Taylor Swift on Writing Her Own Rules, Not Becoming a Cliche and the Hurdle of Going Pop". Billboard. Archived from the original on December 26, 2014. Retrieved February 27, 2019.
Talbott, Chris (October 13, 2013). "Taylor Swift talks next album, CMAs and Ed Sheeran". Associated Press. Archived from the original on October 26, 2013. Retrieved October 26, 2013.
Swift, Taylor (2014). 1989 (CD liner notes). Big Machine Records. BMRBD0500A.
Jo Sales, Nancy; Diehl, Jessica (March 15, 2013). "Taylor Swift's Telltale Heart". Vanity Fair. No. April 2013. Archived from the original on November 1, 2020. Retrieved November 1, 2020.
Hindo, Madison; High, Largo (February 12, 2015). "Taylor Swift Has Reinvented Her Public Image with 1989". Tampa Bay Times. Archived from the original on August 31, 2020. Retrieved August 31, 2020.
Chang, Bee-Shyuan (March 15, 2013). "Taylor Swift Gets Some Mud on Her Boots". The New York Times. Archived from the original on March 22, 2013. Retrieved March 22, 2016.
Eells, Josh (September 8, 2014). "Cover Story: The Reinvention of Taylor Swift". Rolling Stone. Archived from the original on August 16, 2018. Retrieved February 6, 2019.
Yuan, Jada (February 13, 2015). "On the Road with Best Friends Taylor Swift and Karlie Kloss". Vogue. Archived from the original on November 4, 2015. Retrieved November 10, 2015.
Block, Melissa (October 31, 2014). "'Anything That Connects': A Conversation With Taylor Swift" (Audio upload and transcript). NPR. Archived from the original on February 6, 2015. Retrieved January 30, 2015.
Klosterman, Chuck (October 15, 2015). "Taylor Swift on 'Bad Blood', Kanye West, and How People Interpret Her Lyrics". GQ. Archived from the original on October 18, 2015. Retrieved October 18, 2015.
Sloan 2021, p. 23.
Gore, Sydney (November 5, 2014). "Taylor Swift – 1989". The 405. Archived from the original on November 8, 2014. Retrieved November 1, 2020.
Sloan 2021, p. 23; Sloan, Harding & Gottlieb 2019, p. 34.
Dahi, Melissa (November 24, 2014). "Why You Keep Mishearing That Taylor Swift Lyric". The Cut. Archived from the original on February 20, 2021. Retrieved September 17, 2020.
Rosen, Christopher (May 25, 2015). "Even Taylor Swift's Mom Thought It Was 'Starbucks Lovers'". Entertainment Weekly. Archived from the original on October 27, 2020. Retrieved September 17, 2020.
"Taylor Swift Explained to Us the Story and Misconceptions of 'Blank Space'". NME. August 27, 2017. Archived from the original on September 23, 2019. Retrieved August 14, 2020.
Sloan 2021, p. 24.
Levy, Piet (December 12, 2014). "Taylor Swift's 'Blank Space' among the 10 best songs of 2014". Milwaukee Journal Sentinel. Archived from the original on October 20, 2020. Retrieved August 14, 2020.
Baesley, Corey (October 30, 2014). "Taylor Swift: 1989". PopMatters. Archived from the original on March 1, 2019. Retrieved February 4, 2019.
Jagoda, Vrinda (August 19, 2019). "Taylor Swift: 1989 Album Review". Pitchfork. Archived from the original on September 22, 2019. Retrieved September 14, 2019.
Lansky, Sam (October 23, 2014). "Review: 1989 Marks a Paradigm Swift". Time. Archived from the original on October 23, 2014. Retrieved November 17, 2014.
He, Kristen (November 9, 2017). "Why Taylor Swift's 1989 Is Her Best Album: Critic's Take". Billboard. Archived from the original on November 9, 2017. Retrieved November 9, 2017.
Gill, Andy (October 24, 2014). "Taylor Swift, 1989 – Album Review: Pop Star Shows 'Promising Signs of Maturity'". The Independent. Archived from the original on October 31, 2014. Retrieved October 31, 2014.
Zaleski 2024, p. 109.
Wood, Mikael (October 27, 2014). "Review: Taylor Swift Smooths Out the Wrinkles on Sleek 1989". Los Angeles Times. Archived from the original on November 15, 2014. Retrieved November 15, 2014.
Petridis, Alexis (April 26, 2019). "Taylor Swift's Singles – Ranked!". The Guardian. Archived from the original on April 27, 2019. Retrieved August 14, 2020.
Unterberger, Andrew (October 28, 2014). "Taylor Swift Gets Clean, Hits Reset on New Album 1989'". Spin. Archived from the original on November 19, 2018. Retrieved April 5, 2018.
Lipshutz, Jason (October 30, 2014). "Taylor Swift's Next 1989 Single Announced". Billboard. Archived from the original on December 2, 2014. Retrieved October 31, 2014.
"Hot AC | Genres | Republic Playbook". Republic Records. Archived from the original on October 30, 2014. Retrieved October 30, 2014.
"Top 40/M Future Releases". All Access Music Group. Archived from the original on November 30, 2014. Retrieved November 9, 2014.
Mompellio, Gabriel. "Taylor Swift 'Blank Space'". radiodate.it (in Italian). Archived from the original on February 25, 2022. Retrieved February 25, 2022.
"Blank Space (2-Track)". Amazon.de (in German). Archived from the original on January 7, 2015. Retrieved May 11, 2015.
Trust, Gary (November 5, 2014). "Taylor Swift's 'Shake It Off' Returns to No. 1 on Hot 100". Billboard. Archived from the original on November 20, 2015. Retrieved November 5, 2014.
Trust, Gary (November 19, 2014). "Taylor Swift Makes Hot 100 History With 'Blank Space'". Billboard. Archived from the original on February 18, 2017. Retrieved November 19, 2014.
Trust, Gary (December 31, 2014). "Taylor Swift Helps Tie Record Streak for Women Atop Hot 100". Billboard. Archived from the original on February 3, 2015. Retrieved August 15, 2020.
Zellner, Xander (August 14, 2023). "Taylor Swift's 'Blank Space' Returns to Hot 100 for First Time in Eight Years". Billboard. Archived from the original on August 16, 2023. Retrieved August 15, 2023.
Zellner, Xander (August 22, 2023). "Taylor Swift Extends Artist 100 Longevity Record to 77 Weeks at No. 1". Billboard. Archived from the original on August 22, 2023. Retrieved August 22, 2023.
"American single certifications – Taylor Swift – Blank Space". Recording Industry Association of America. Retrieved June 2, 2020.
Trust, Gary (October 21, 2022). "Ask Billboard: Taylor Swift's Career Streaming, Airplay & Sales, Ahead of the Chart Debut of Midnights". Billboard. Archived from the original on October 21, 2022. Retrieved July 24, 2024.
"Taylor Swift – Blank Space". ARIA Top 50 Singles. Retrieved November 22, 2014.
"Taylor Swift Chart History (Canadian Hot 100)". Billboard. Retrieved November 20, 2014.
"EMA Top 10 Airplay: Week Ending January 20, 2015". Entertainment Monitoring Africa. Archived from the original on January 23, 2015. Retrieved December 7, 2020.
"Official Scottish Singles Sales Chart Top 100". Official Charts Company. Retrieved December 7, 2014.
"Taylor Swift Chart History (Euro Digital Song Sales)". Billboard. Retrieved December 11, 2014.
"Taylor Swift: Blank Space" (in Finnish). Musiikkituottajat. Retrieved January 8, 2015.
"Taylor Swift – Blank Space". Top 40 Singles. Retrieved November 28, 2014.
"Listy bestsellerów, wyróżnienia :: Związek Producentów Audio-Video". Polish Airplay Top 100. Retrieved February 2, 2015.
"ČNS IFPI" (in Slovak). Hitparáda – Radio Top 100 Oficiálna. IFPI Czech Republic. Note: insert 201512 into search. Retrieved March 23, 2015.
"Airplay Top5 – 09.02.2015 – 15.02.2015"". Bulgarian Association of Music Producers. Archived from the original on February 21, 2015. Retrieved February 9, 2015.
"ČNS IFPI" (in Czech). Hitparáda – Radio Top 100 Oficiální. IFPI Czech Republic. Note: Select 07. týden 2015 in the date selector. Retrieved February 16, 2015.
"The Irish Charts – Search Results – Taylor Swift". Irish Singles Chart. Retrieved January 28, 2020.
"Media Forest Week 50, 2014". Israeli Airplay Chart. Media Forest. Retrieved December 16, 2014.
"Official Singles Chart Top 100". Official Charts Company. Retrieved December 7, 2014.
"The official lebanese Top 20 – Taylor Swift". The Official Lebanese Top 20. Archived from the original on September 17, 2016. Retrieved September 30, 2017.
"ARIA Charts – Accreditations – 2024 Singles" (PDF). Australian Recording Industry Association. Retrieved June 30, 2024.
"Brazilian single certifications – Taylor Swift – Blank Space" (in Portuguese). Pro-Música Brasil. Retrieved July 22, 2024.
"Canadian single certifications – Taylor Swift – Blank Space". Music Canada. Retrieved December 1, 2015.
"Wyróżnienia – Platynowe płyty CD - Archiwum - Przyznane w 2021 roku" (in Polish). Polish Society of the Phonographic Industry. Retrieved August 11, 2021.
"New Zealand single certifications – Taylor Swift – Blank Space". Radioscope. Retrieved December 19, 2024. Type Blank Space in the "Search:" field.
"British single certifications – Taylor Swift – Blank Space". British Phonographic Industry. Retrieved May 12, 2023.
"Austrian single certifications – Taylor Swift – Blank Space" (in German). IFPI Austria. Retrieved May 29, 2024.
"Portuguese single certifications – Taylor Swift – Blank Space" (PDF) (in Portuguese). Associação Fonográfica Portuguesa. Retrieved August 23, 2023.
"Global Music Report 2016" (PDF). International Federation of the Phonographic Industry. April 12, 2016. Archived from the original (PDF) on July 30, 2016. Retrieved April 16, 2016.
Kimberlin, Shane (November 3, 2014). "Taylor Swift – 1989 | Album Review". musicOMH. Archived from the original on November 5, 2014. Retrieved February 5, 2019.
Cliff, Aimee (October 30, 2014). "1989". Fact. Archived from the original on June 18, 2020. Retrieved August 14, 2020.
Leedham, Robert (October 30, 2014). "Album Review: Taylor Swift – 1989". Drowned in Sound. Archived from the original on February 14, 2019. Retrieved August 14, 2020.
Empire, Kitty (October 26, 2014). "Taylor Swift: 1989 review – a bold, gossipy confection". The Observer. Archived from the original on October 26, 2014. Retrieved April 6, 2019.
Caramanica, Jon (October 23, 2014). "Review: Taylor Swift's 1989". The New York Times. Archived from the original on October 26, 2014. Retrieved October 23, 2014.
Sheffield, Rob (December 12, 2019). "Taylor Swift's Songs: All Ranked". Rolling Stone. Archived from the original on October 21, 2020. Retrieved August 5, 2020.
"All 158 Taylor Swift Songs, Ranked". Paste. February 11, 2020. Archived from the original on April 13, 2020. Retrieved August 15, 2020.
Rankin, Selje (July 31, 2020). "The best song from every Taylor Swift album". Entertainment Weekly. Archived from the original on October 26, 2020. Retrieved August 15, 2020.
Lee, Taila (January 26, 2023). "The Taylor Swift Essentials: 13 Songs That Display Her Storytelling Prowess And Genre-Bouncing Genius". The Recording Academy. Archived from the original on April 14, 2023. Retrieved November 1, 2023.
"50 Best Songs Of 2014". Rolling Stone. December 3, 2014. Archived from the original on December 25, 2014. Retrieved December 3, 2014.
"The 100 Best Songs of the 2010s". Rolling Stone. December 4, 2019. Archived from the original on December 6, 2019. Retrieved December 6, 2019.
"The 500 Greatest Songs of All Time". Rolling Stone. September 15, 2021. Archived from the original on September 15, 2021. Retrieved July 18, 2022.
"The 500 Greatest Songs of All Time". Rolling Stone. February 16, 2024. Archived from the original on February 16, 2024. Retrieved February 18, 2024.
"Top 10 songs". Time. December 2, 2014. Archived from the original on January 24, 2017. Retrieved December 2, 2014.
Young, Alex (January 13, 2015). "According to 600 music critics, these were the best albums and songs of 2014". Consequence of Sound. Archived from the original on October 19, 2020. Retrieved August 14, 2020.
"The 200 Best Songs of the 2010s". Stereogum. November 5, 2019. Archived from the original on November 6, 2019. Retrieved November 23, 2019.
"All The Best Songs of the 2010s, Ranked". Uproxx. October 9, 2019. Archived from the original on April 17, 2020. Retrieved December 11, 2019.
Atkinson, Katie (November 21, 2019). "Songs That Defined the Decade: Taylor Swift's 'Blank Space'". Billboard. Archived from the original on November 29, 2019. Retrieved November 23, 2019.
"The 100 Best Singles of the 2010s". Slant Magazine. January 1, 2020. Archived from the original on January 1, 2020. Retrieved January 1, 2020.
"American Music Awards 2015: Full Winners List". Variety. November 22, 2015. Archived from the original on June 21, 2019. Retrieved August 15, 2020.
"BMI Honors Taylor Swift and Legendary Songwriting Duo Mann & Weil at the 64th Annual BMI Pop Awards". Broadcast Music, Inc. May 11, 2016. Archived from the original on June 2, 2016. Retrieved June 2, 2016.
"APRA Music Awards 2016". APRA AMCOS. April 5, 2016. Archived from the original on March 20, 2024. Retrieved April 8, 2024.
"Grammy awards winners: the full list". The Guardian. February 16, 2016. Archived from the original on February 21, 2016. Retrieved December 14, 2016.
Anthony Hernandez, Brian (November 10, 2014). "Taylor Swift Cries, Screams, Stabs In Crazy 'Blank Space' Music Video". Mashable. Archived from the original on May 7, 2021. Retrieved August 14, 2020.
Spanos, Brittany (November 11, 2014). "Taylor Swift's 'Blank Space' Director Details Interactive App". Rolling Stone. Archived from the original on November 12, 2014. Retrieved November 11, 2014.
Sullivan, Kevin P. (November 14, 2014). "Taylor Swift Really Stood on a Horse And 7 Other Secrets From 'Blank Space'". MTV. Archived from the original on November 17, 2014. Retrieved November 16, 2014.
Roberts, Randall (November 10, 2014). "Taylor Swift Loses Mind, Smashes, Slashes in New 'Blank Space' Clip". Los Angeles Times. Archived from the original on November 29, 2014. Retrieved August 16, 2020.
McKinney, Kelsey (November 10, 2014). "Taylor Swift's New Music Video Is Back Online". Vox. Archived from the original on November 12, 2014. Retrieved January 11, 2024.
"Taylor and American Express Brings Fans a First-of-Its-Kind Video Experience for 'Blank Space' Music Video". taylorswift.com. Archived from the original on November 14, 2014. Retrieved November 14, 2014.
Lipshutz, Jason (November 11, 2014). "Taylor Swift's Blank Space App: Inside The User Experience". Billboard. Archived from the original on September 29, 2020. Retrieved September 15, 2020.
Zuckerman, Esther (November 10, 2014). "Taylor Swift Is In On the Joke in Her 'Blank Space' Video". Entertainment Weekly. Archived from the original on January 4, 2020. Retrieved January 4, 2020.
"Taylor Swift's Top 10 Best Music Videos". Billboard. August 30, 2015. Archived from the original on December 1, 2015. Retrieved December 1, 2015.
Valenti, Jessica (November 11, 2014). "Taylor Swift in the 'Blank Space' Video Is the Woman We've Been Waiting For". The Guardian. Archived from the original on March 29, 2020. Retrieved March 29, 2020.
Hutcheson, Susannah (August 31, 2017). "The Top 10 Taylor Swift Music Videos, Ranked". USA Today. Archived from the original on July 7, 2019. Retrieved August 31, 2017.
"30 Taylor Swift Music Videos, Ranked". Spin. November 12, 2017. Archived from the original on November 13, 2017. Retrieved November 12, 2017.
Rankin, Seija; Huff, Lauren (June 30, 2020). "The Best Taylor Swift Music Video From Every Album". Entertainment Weekly. Archived from the original on July 3, 2020. Retrieved June 30, 2020.
Lipshutz, Jason (August 30, 2015). "MTV Video Music Awards 2015: The Winners". Billboard. Archived from the original on July 1, 2020. Retrieved July 1, 2020.
"MTV Video Music Awards Japan 2015". MTV Japan. Archived from the original on February 21, 2016. Retrieved February 28, 2016.
Li, Shirley (September 10, 2015). "Emmys 2015: Taylor Swift wins first Emmy, more early honors announced". Entertainment Weekly. Archived from the original on October 28, 2019. Retrieved October 28, 2019.
"The 100 Greatest Music Videos". Rolling Stone. July 30, 2021. Archived from the original on July 30, 2021. Retrieved July 31, 2021.
Stutz, Colin (October 27, 2014). "Taylor Swift Live-Broadcasts Manhattan Rooftop Secret Session". The Hollywood Reporter. Archived from the original on October 31, 2014. Retrieved February 6, 2019.
"Watch Taylor Swift's Theatrical 'Blank Space' Live Debut at AMAs". Rolling Stone. November 24, 2014. Archived from the original on September 3, 2017. Retrieved November 24, 2014.
Benjamin, Jeff (November 27, 2014). "Taylor Swift Brings Dramatic Flair to "Blank Space" Live on 'The Voice'". Fuse. Archived from the original on November 7, 2017. Retrieved May 29, 2019.
Harvey, Lydia (December 3, 2014). "Taylor Swift prances around in lingerie during Victoria's Secret Fashion Show". Tampa Bay Times. Archived from the original on February 6, 2019.
Stutz, Colin (December 6, 2014). "Taylor Swift Beats Laryngitis, Sam Smith, Ariana Grande Shine at KIIS FM Jingle Ball". Billboard. Archived from the original on December 8, 2014.
Lee, Ashley (February 25, 2015). "Brit Awards 2015: Taylor Swift Performs 'Blank Space'". The Hollywood Reporter. Archived from the original on January 29, 2021. Retrieved January 27, 2021.
Yahr, Emily (May 5, 2015). "Taylor Swift '1989' World Tour: Set list, costumes, the stage, the spectacle". The Washington Post. Archived from the original on October 12, 2018. Retrieved December 12, 2018.
Sheffield, Rob (May 9, 2018). "Why Taylor Swift's 'Reputation' Tour Is Her Finest Yet". Rolling Stone. Archived from the original on September 12, 2018. Retrieved December 12, 2018.
Shafer, Ellise (March 18, 2023). "Taylor Swift Eras Tour: The Full Setlist From Opening Night". Variety. Archived from the original on March 18, 2023. Retrieved March 19, 2023.
Mylrea, Hannah (September 10, 2019). "Taylor Swift's The City of Lover concert: a triumphant yet intimate celebration of her fans and career". NME. Archived from the original on September 16, 2019. Retrieved September 12, 2019.
Aniftos, Rania (October 20, 2019). "Taylor Swift, Billie Eilish & More Supported a Great Cause at 7th Annual We Can Survive Concert: Recap". Billboard. Archived from the original on October 22, 2019. Retrieved October 23, 2019.
Gracie, Bianca (November 24, 2019). "Taylor Swift Performs Major Medley Of Hits, Brings Out Surprise Guests For 'Shake It Off' at 2019 AMAs". Billboard. Archived from the original on November 26, 2019. Retrieved November 25, 2019.
Iasimone, Ashley (December 8, 2019). "Taylor Swift Performs 'Christmas Tree Farm' Live for the First Time at Capital FM's Jingle Bell Ball: Watch". Billboard. Archived from the original on December 8, 2019. Retrieved December 9, 2019.
Mastrogiannis, Nicole (December 14, 2019). "Taylor Swift Brings Holiday Cheer to Jingle Ball with "Christmas Tree Farm"". iHeartMedia. Archived from the original on December 14, 2019. Retrieved December 14, 2019.
Menyes, Carolyn (December 15, 2014). "Pitbull Reworks Taylor Swift 'Blank Space': Listen to the AMAs Inspired Mr. Worldwide Remix". Music Times. Archived from the original on April 25, 2018. Retrieved April 25, 2018.
"Taylor Swift's Song 'Blank Space' Goes Viral With 1940's Style YouTube Cover". Capital FM. December 17, 2014. Archived from the original on May 17, 2021. Retrieved September 15, 2020.
Trudon, Taydor (February 13, 2015). "Imagine Dragons Jam Out To Taylor Swift's 'Blank Space' In Soulful Cover". HuffPost. Archived from the original on October 21, 2020. Retrieved August 17, 2020.
"Blank Space –Single by I Prevail". Apple Music. December 5, 2014. Archived from the original on October 18, 2020. Retrieved October 18, 2020.
White, Emily (June 16, 2015). "I Prevail's Punk Cover of Taylor Swift's 'Blank Space' Debuts on Hot Rock Songs". Billboard. Archived from the original on September 29, 2015. Retrieved October 12, 2015.
"I Prevail Chart History". Billboard. Archived from the original on July 16, 2018. Retrieved July 23, 2018.
"American certifications – I Prevail – Blank Space". Recording Industry Association of America. Retrieved April 24, 2021.
Young, Alex (September 18, 2015). "Ryan Adams has finished his Taylor Swift 1989 covers album, preview two songs". Consequence of Sound. Archived from the original on September 27, 2015. Retrieved September 22, 2015.
Winograd, Jeremy (October 21, 2015). "Review: Ryan Adams, 1989". Slant Magazine. Archived from the original on May 6, 2019. Retrieved August 17, 2020.
Zaleski, Annie (September 21, 2015). "Ryan Adams transforms Taylor Swift's 1989 into a melancholy masterpiece". The A.V. Club. Archived from the original on February 12, 2018. Retrieved February 12, 2018.
Young, Alex (September 22, 2015). "Father John Misty covers Ryan Adams' cover of Taylor Swift's "Blank Space" (in the style of Velvet Underground)". Consequence of Sound. Archived from the original on September 22, 2015. Retrieved September 22, 2015.
Gordon, Jeremy (September 21, 2015). "Father John Misty Takes on Ryan Adams, Covers Taylor Swift's 'Blank Space' in the Spirit of the Velvet Underground". Pitchfork. Archived from the original on September 23, 2015. Retrieved September 22, 2015.
"Taylor Swift – Blank Space" (in German). Ö3 Austria Top 40. Retrieved November 26, 2014.
"Taylor Swift – Blank Space" (in Dutch). Ultratop 50. Retrieved January 17, 2015.
"Taylor Swift – Blank Space" (in French). Ultratop 50. Retrieved January 17, 2015.
"Hot 100 Billboard Brasil – weekly". Billboard Brasil. January 17, 2014. Archived from the original on January 22, 2015. Retrieved May 13, 2015.
"Taylor Swift Chart History (Canada AC)". Billboard. Retrieved March 30, 2015.
"Taylor Swift Chart History (Canada CHR/Top 40)". Billboard. Retrieved December 11, 2014.
"Taylor Swift Chart History (Canada Hot AC)". Billboard. Retrieved March 30, 2015.
Taylor Swift — Blank Space. TopHit. Retrieved December 7, 2020.
"Taylor Swift: Blank Space" (in Finnish). Musiikkituottajat. Retrieved June 25, 2019.
"Taylor Swift – Blank Space" (in French). Les classement single. Retrieved March 21, 2015.
"Top de la semaine". snepmusique.com. Retrieved July 6, 2024.
"Taylor Swift – Blank Space" (in German). GfK Entertainment charts. Retrieved November 24, 2014.
"Taylor Swift Album & Chart History". Billboard Greece Digital Songs for Taylor Swift. Archived from the original on July 25, 2015. Retrieved December 4, 2014.
"Archívum – Slágerlisták – MAHASZ" (in Hungarian). Single (track) Top 40 lista. Magyar Hanglemezkiadók Szövetsége. Retrieved January 29, 2015.
"Taylor Swift Chart History". RÚV. April 11, 2016. Archived from the original on September 3, 2017. Retrieved May 28, 2017.
"Top Digital Download – Classifica settimanale WK 3 (dal 2015-01-12 al 2015-01-18)" (in Italian). Federazione Industria Musicale Italiana. Archived from the original on January 22, 2015. Retrieved January 15, 2014.
"Taylor Swift Chart History (Italy Digital Song Sales)". Billboard. Archived from the original on June 29, 2022. Retrieved August 11, 2021.
"Taylor Swift Chart History (Japan Hot 100)". Billboard. Retrieved December 4, 2014.
"Japan Adult Contemporary Airplay Chart". Billboard Japan (in Japanese). Archived from the original on February 27, 2024. Retrieved October 31, 2023.
"Taylor Swift Chart History (Luxembourg Digital Song Sales)". Billboard. Retrieved December 4, 2014. [dead link]
"Mexico Airplay". Billboard. January 2, 2013. Archived from the original on September 3, 2015. Retrieved February 12, 2015.
"Top 20 Inglés Del 2 al 8 de Febrero, 2015". Monitor Latinoaccessdate=2018-05-02. February 2, 2015.
"Nederlandse Top 40 – Taylor Swift" (in Dutch). Dutch Top 40. Retrieved January 11, 2015.
"Taylor Swift - Blank Space". Dutch Top 40 (in Dutch). Archived from the original on May 13, 2022. Retrieved May 13, 2022.
"Taylor Swift Chart History (Portugal Digital Song Sales)". Billboard. Retrieved August 11, 2021. [dead link]
"Airplay 100 – 1 februarie 2015" (in Romanian). Kiss FM. February 1, 2015. Archived from the original on July 29, 2020. Retrieved April 9, 2020.
"SloTop50 | Slovenian official singles weekly charts" (in Slovenian). Archived from the original on August 28, 2017. Retrieved February 5, 2018.
"2014년 48주차 Digital Chart" (in Korean). Gaon Music Chart. Archived from the original on December 19, 2014. Retrieved December 14, 2014.
"Taylor Swift – Blank Space" Canciones Top 50. Retrieved January 11, 2015.
"Taylor Swift – Blank Space". Swiss Singles Chart. Retrieved February 1, 2015.
"19, 2015 Ukraine Airplay Chart for January 19, 2015." TopHit. Retrieved December 7, 2020.
"Taylor Swift Chart History (Hot 100)". Billboard. Retrieved December 20, 2023.
"Taylor Swift Chart History (Adult Contemporary)". Billboard. Retrieved February 12, 2015.
"Taylor Swift Chart History (Adult Pop Songs)". Billboard. Retrieved November 20, 2014.
"Taylor Swift Chart History (Dance Club Songs)". Billboard. Retrieved December 28, 2014.
"Taylor Swift Chart History (Latin Airplay)". Billboard. Retrieved March 10, 2015.
"Taylor Swift Chart History (Pop Songs)". Billboard. Retrieved November 20, 2014.
"Taylor Swift Chart History (Rhythmic)". Billboard. Retrieved November 20, 2014.
"Record Report - Rock General" (in Spanish). Record Report. Archived from the original on December 16, 2014. Retrieved January 29, 2014.
"Taylor Swift Chart History (Global 200)". Billboard. Retrieved December 20, 2023.
"IFPI Charts – Digital Singles Chart (International) – Εβδομάδα: 43/2023" (in Greek). IFPI Greece. Retrieved November 7, 2023.
"Classifica settimanale WK 44" (in Italian). Federazione Industria Musicale Italiana. Archived from the original on March 18, 2020. Retrieved November 5, 2023.
"TOP 20 Most Streamed International Singles In Malaysia Week 10 (01/03/2024-07/03/2024)". RIM. March 16, 2024. Archived from the original on March 16, 2024. Retrieved March 16, 2024 – via Facebook.
"Dutch Single Tip 22/07/2023". dutchcharts.nl (in Dutch). Archived from the original on July 25, 2023. Retrieved July 24, 2023.
"NZ Top 40 Singles Chart". Recorded Music NZ. Archived from the original on July 8, 2023. Retrieved July 26, 2023.
"Taylor Swift Chart History (Philippines Songs)". Billboard. Archived from the original on November 8, 2023. Retrieved July 11, 2023.
"OLiS – oficjalna lista airplay" (Select week 06.01.2024–12.01.2024.) (in Polish). OLiS. Archived from the original on October 23, 2023. Retrieved January 15, 2024.
"Taylor Swift – Blank Space". AFP Top 100 Singles. Retrieved November 9, 2023.
"RIAS Top Charts Week 44 (27 Oct – 2 Nov 2023)". RIAS. Archived from the original on November 7, 2023. Retrieved November 7, 2023.
"Veckolista Heatseeker, vecka 28". Sverigetopplistan. Archived from the original on July 28, 2023. Retrieved July 28, 2023.
"Official Audio Streaming Chart Top 100". Official Charts Company. Retrieved November 3, 2023.
"Hot 100: Week of August 26, 2023". Billboard. Archived from the original on September 1, 2023. Retrieved August 22, 2023.
"Taylor Swift Chart History (Billboard Vietnam Hot 100)". Billboard. Archived from the original on June 6, 2022. Retrieved July 14, 2023.
"End of Year Charts – ARIA Top 100 Singles 2014". Australian Recording Industry Association. Archived from the original on January 9, 2015. Retrieved February 4, 2015.
"Single Top 100 – eladási darabszám alapján – 2014" (in Hungarian). Mahasz. Archived from the original on June 3, 2020. Retrieved November 11, 2019.
"Top Selling Singles of 2014". Recorded Music NZ. Archived from the original on March 6, 2016. Retrieved December 27, 2014.
"The Official Top 100 Biggest Songs of 2014 revealed". Official Charts Company. Archived from the original on February 8, 2015. Retrieved December 31, 2014.
"ARIA Charts – End of Year Charts – Top 100 Singles 2015". Australian Recording Industry Association. Archived from the original on January 24, 2016. Retrieved January 6, 2016.
"Jahreshitparade Singles 2015". Hung Medien. Archived from the original on February 17, 2020. Retrieved January 8, 2020.
"Jaaroverzichten 2015". Ultratop. Archived from the original on August 19, 2016. Retrieved November 11, 2019.
"Rapports Annuels 2015". Ultratop. Archived from the original on August 17, 2016. Retrieved November 11, 2019.
"Canadian Hot 100 Year End 2015". Billboard. Archived from the original on December 11, 2015. Retrieved December 9, 2015.
"Top de l'année Top Singles 2015" (in French). Syndicat National de l'Édition Phonographique. Archived from the original on July 3, 2020. Retrieved August 8, 2020.
"Top 100 Single – Jahrescharts" (in German). GfK Entertainment. Archived from the original on December 22, 2016. Retrieved January 6, 2016.
"Single Top 100 – eladási darabszám alapján – 2015" (in Hungarian). Mahasz. Archived from the original on June 3, 2020. Retrieved November 11, 2019.
"Top 100 – Jaaroverzicht van 2015". Dutch Top 40. Archived from the original on May 20, 2019. Retrieved November 11, 2019.
"Jaaroverzichten - 3FM 2015". Dutch Charts (in Dutch). Archived from the original on May 13, 2022. Retrieved May 13, 2022.
"Airplay – podsumowanie 2015 roku" (in Polish). Polish Society of the Phonographic Industry. Archived from the original on February 11, 2017. Retrieved December 30, 2015.
"SloTop50: Slovenian official year end singles chart". slotop50.si. Archived from the original on March 3, 2016. Retrieved July 18, 2015.
"Schweizer Jahreshitparade 2015 – hitparade.ch". Hung Medien. Archived from the original on September 13, 2019. Retrieved November 11, 2019.
"End of Year Singles Chart Top 100 – 2015". Official Charts Company. Archived from the original on February 12, 2016. Retrieved January 5, 2016.
"Hot 100: Year End 2015". Billboard. Archived from the original on August 23, 2016. Retrieved December 9, 2015.
"Adult Contemporary Songs Year End 2015". Billboard. Archived from the original on December 11, 2015. Retrieved December 9, 2015.
"Adult Pop Songs Year End 2015". Billboard. Archived from the original on December 14, 2015. Retrieved December 9, 2015.
"Pop Songs Year End 2015". Billboard. Archived from the original on December 21, 2015. Retrieved December 9, 2015.
"As 100 Mais Tocadas nas Rádios Jovens em 2016". Billboard Brasil (in Portuguese). January 4, 2017. Archived from the original on September 7, 2017. Retrieved September 7, 2017.
"ARIA Top 100 Singles Chart for 2023". Australian Recording Industry Association. Archived from the original on January 12, 2024. Retrieved January 12, 2024.
"Billboard Global 200 – Year-End 2023". Billboard. Archived from the original on November 21, 2023. Retrieved November 21, 2023.
"2019 ARIA End of Decade Singles Chart". ARIA Charts. January 2020. Archived from the original on January 11, 2020. Retrieved January 17, 2020.
"Decade-End Charts: Hot 100 Songs". Billboard. Archived from the original on November 14, 2019. Retrieved November 15, 2019.
"Greatest of All Time Hot 100 Songs". Billboard. Archived from the original on August 3, 2018. Retrieved October 1, 2023.
"Greatest of All Time Pop Songs". Billboard. Archived from the original on June 13, 2018. Retrieved January 8, 2021.
"Danish single certifications – Taylor Swift – Blank Space". IFPI Danmark. May 16, 2023. Retrieved May 16, 2023.
"Gold-/Platin-Datenbank (Taylor Swift; 'Blank Space')" (in German). Bundesverband Musikindustrie. Retrieved June 7, 2024.
"Italian single certifications – Taylor Swift – Blank Space" (in Italian). Federazione Industria Musicale Italiana. Retrieved February 25, 2019. Select "2019" in the "Anno" drop-down menu. Type "Blank Space" in the "Filtra" field. Select "Singoli" under "Sezione".
"Japanese digital single certifications – Taylor Swift – Blank Space" (in Japanese). Recording Industry Association of Japan. Retrieved September 9, 2021. Select 2018年7月 on the drop-down menu
"Certificaciones" (in Spanish). Asociación Mexicana de Productores de Fonogramas y Videogramas. Retrieved November 1, 2020. Type Taylor Swift in the box under the ARTISTA column heading and Blank Space in the box under the TÍTULO column heading.
"Norwegian single certifications – Taylor Swift – Blank Space" (in Norwegian). IFPI Norway. Retrieved October 29, 2020.
"Spanish single certifications – Taylor Swift – Blank Space". El portal de Música. Productores de Música de España. Retrieved April 1, 2024.
"The Official Swiss Charts and Music Community: Awards ('Blank Space')". IFPI Switzerland. Hung Medien. Retrieved November 1, 2020.
"Japanese single streaming certifications – Taylor Swift – Blank Space" (in Japanese). Recording Industry Association of Japan. Retrieved November 27, 2024. Select 2024年10月 on the drop-down menu
Melas, Chloe (November 16, 2020). "Taylor Swift Speaks Out about Sale of Her Masters". CNN. Archived from the original on November 18, 2020. Retrieved November 19, 2020.
"Taylor Swift Wants to Re-Record Her Old Hits". BBC News. August 22, 2019. Archived from the original on August 22, 2019. Retrieved August 22, 2019.
Finnis, Alex (November 17, 2020). "Taylor Swift Masters: The Controversy around Scooter Braun Selling the Rights to Her Old Music Explained". i. Archived from the original on February 12, 2021. Retrieved February 13, 2021.
Shah, Neil (April 9, 2021). "Taylor Swift Releases New Fearless Album, Reclaiming Her Back Catalog". The Wall Street Journal. Archived from the original on October 8, 2021. Retrieved September 25, 2022.
Vassell, Nicole (October 27, 2023). "Taylor Swift Fans Celebrate As Pop Star Releases 1989 (Taylor's Version)". The Independent. Archived from the original on October 30, 2023. Retrieved October 30, 2023.
D'Souza, Shaad (October 30, 2023). "Taylor Swift: 1989 (Taylor's Version) Album Review". Pitchfork. Archived from the original on October 30, 2023. Retrieved October 30, 2023.
Swift, Taylor (2023). 1989 (Taylor's Version) (Compact disc liner notes). Republic Records. 0245597656.
Barnes, Kelsey (October 27, 2023). "Taylor Swift: 1989 (Taylor's Version)". The Line of Best Fit. Archived from the original on October 27, 2023. Retrieved October 27, 2023.
White, Adam (October 27, 2023). "Taylor Swift Re-Records Her Pop Classic 1989 to Diminishing Returns – Review". The Independent. Archived from the original on October 27, 2023. Retrieved October 27, 2023.
Geraghty, Hollie (October 27, 2023). "Taylor Swift – 1989 (Taylor's Version) Review: Her Best Album Will Never Go Out of Style". NME. Archived from the original on October 27, 2023. Retrieved October 27, 2023.
Trust, Gary (November 6, 2023). "Taylor Swift Makes History With Top 6 Songs, All From 1989 (Taylor's Version), on Billboard Global 200 Chart". Billboard. Archived from the original on November 14, 2023. Retrieved December 20, 2023.
"ARIA Top 50 Singles Chart". Australian Recording Industry Association. November 6, 2023. Archived from the original on November 3, 2023. Retrieved November 3, 2023.
"NZ Top 40 Singles Chart". Recorded Music NZ. November 6, 2023. Archived from the original on November 3, 2023. Retrieved November 4, 2023.
Zellner, Xander (November 6, 2023). "Taylor Swift Charts All 21 Songs From 1989 (Taylor's Version) on the Hot 100". Billboard. Archived from the original on November 6, 2023. Retrieved December 20, 2023.
"Taylor Swift Chart History (Brasil Hot 100)". Billboard. Retrieved November 9, 2023.
"Taylor Swift – Blank Space (Taylor's Version)" (in French). Les classement single. Retrieved November 16, 2023.
"Digital Singles Chart (International)". IFPI Greece. Archived from the original on November 13, 2023. Retrieved November 8, 2023.
"Taylor Swift Chart History (Ireland Songs)". Billboard. Archived from the original on November 7, 2023. Retrieved November 7, 2023.
"Taylor Swift Chart History (Malaysia Songs)". Billboard. Archived from the original on May 17, 2022. Retrieved November 7, 2023.
"TOP 20 Most Streamed International Singles In Malaysia Week 44 (27/10/2023- 02/11/2023)". RIM. November 11, 2023. Archived from the original on November 12, 2023. Retrieved November 12, 2023 – via Facebook.
"This Week's Official MENA Chart Top 20: from 27/10/2023 to 02/11/2023". International Federation of the Phonographic Industry. October 27, 2023. Archived from the original on November 8, 2023. Retrieved November 8, 2023.
"Taylor Swift – Blank Space (Taylor's Version)" Canciones Top 50. Retrieved November 17, 2023.
"Taylor Swift – Blank Space (Taylor's Version)". Singles Top 100. Retrieved November 17, 2023.
"This Week's Official UAE Chart Top 20: from 27/10/2023 to 02/11/2023". International Federation of the Phonographic Industry. October 27, 2023. Archived from the original on November 8, 2023. Retrieved November 8, 2023.
"Taylor Swift Chart History (U.K. Songs)". Billboard. Archived from the original on November 7, 2023. Retrieved November 7, 2023.
"Official Singles Downloads Chart Top 100". Official Charts Company. Retrieved November 3, 2023.
"Official Singles Sales Chart Top 100". Official Charts Company. Archived from the original on November 11, 2023. Retrieved November 3, 2023.
"Official Streaming Chart Top 100". Official Charts Company. Archived from the original on November 3, 2023. Retrieved November 3, 2023.
"Taylor Swift Chart History (Billboard Vietnam Hot 100)". Billboard. Archived from the original on June 6, 2022. Retrieved November 11, 2023.
"Brazilian single certifications – Taylor Swift – Blank Space (Taylor's Version)" (in Portuguese). Pro-Música Brasil. Retrieved July 24, 2024.
"New Zealand single certifications – Taylor Swift – Blank Space (Taylor's Version)". Radioscope. Retrieved December 19, 2024. Type Blank Space (Taylor's Version) in the "Search:" field.

    "British single certifications – Taylor Swift – Blank Space (Taylor's Version)". British Phonographic Industry. Retrieved August 23, 2024.

Sources

    Sloan, Nate; Harding, Charlie; Gottlieb, Iris (2019). "A Star's Melodic Signature: Melody: Taylor Swift—'You Belong with Me'". Switched on Pop: How Popular Music Works, and Why it Matters. Oxford University Press. pp. 21–35. ISBN 978-0-19-005668-1.
    Sloan, Nate (2021). "Taylor Swift and the Work of Songwriting". Contemporary Music Review. 40 (1): 11–26. doi:10.1080/07494467.2021.1945226. S2CID 237695045.
    Zaleski, Annie (2024). "The 1989 Era". Taylor Swift: The Stories Behind the Songs. Thunder Bay Press. pp. 106–131. ISBN 978-1-6672-0845-9.

    vte

Taylor Swift songs

    Singles discography

Taylor Swift	

    "Tim McGraw" "Picture to Burn" "Teardrops on My Guitar" "A Place in This World" "Should've Said No" "Our Song"

Fearless	

    "Fearless" "Fifteen" "Love Story" "Hey Stephen" "White Horse" "You Belong with Me" "Breathe" "You're Not Sorry" "The Way I Loved You" "Forever & Always" "The Best Day" "Change"

Taylor's Version	

    "You All Over Me" "Mr. Perfectly Fine" "That's When"

Speak Now	

    "Mine" "Sparks Fly" "Back to December" "Speak Now" "Dear John" "Mean" "The Story of Us" "Never Grow Up" "Enchanted" "Better than Revenge" "Innocent" "Haunted" "Long Live" "Ours"

Taylor's Version	

    "Electric Touch" "When Emma Falls in Love" "I Can See You" "Castles Crumbling"

Red	

    "State of Grace" "Red" "Treacherous" "I Knew You Were Trouble" "All Too Well" "22" "We Are Never Ever Getting Back Together" "The Last Time" "Holy Ground" "Everything Has Changed" "Begin Again"

Taylor's Version	

    "Nothing New" "Message in a Bottle" "I Bet You Think About Me" "Forever Winter"

1989	

    "Welcome to New York" "Blank Space" "Style" "Out of the Woods" "All You Had to Do Was Stay" "Shake It Off" "I Wish You Would" "Bad Blood" "Wildest Dreams" "How You Get the Girl" "This Love" "I Know Places" "Clean" "You Are in Love" "New Romantics"

Taylor's Version	

    "'Slut!'" "Say Don't Go" "Now That We Don't Talk" "Suburban Legends" "Is It Over Now?"

Reputation	

    "...Ready for It?" "End Game" "I Did Something Bad" "Don't Blame Me" "Delicate" "Look What You Made Me Do" "So It Goes..." "Gorgeous" "Getaway Car" "Dress" "Call It What You Want" "New Year's Day"

Lover	

    "I Forgot That You Existed" "Cruel Summer" "Lover" "The Man" "The Archer" "Miss Americana & the Heartbreak Prince" "Paper Rings" "Cornelia Street" "Death by a Thousand Cuts" "London Boy" "Soon You'll Get Better" "False God" "You Need to Calm Down" "Me!"

Folklore	

    "The 1" "Cardigan" "The Last Great American Dynasty" "Exile" "My Tears Ricochet" "Mirrorball" "Seven" "August" "This Is Me Trying" "Illicit Affairs" "Invisible String" "Mad Woman" "Epiphany" "Betty" "Peace" "Hoax" "The Lakes"

Evermore	

    "Willow" "Champagne Problems" "'Tis the Damn Season" "Tolerate It" "No Body, No Crime" "Happiness" "Dorothea" "Coney Island" "Long Story Short" "Marjorie"

Midnights	

    "Lavender Haze" "Maroon" "Anti-Hero" "Snow on the Beach" "You're on Your Own, Kid" "Midnight Rain" "Question...?" "Vigilante Shit" "Bejeweled" "Labyrinth" "Karma" "Sweet Nothing" "Mastermind" "Hits Different" "Bigger Than the Whole Sky" "Would've, Could've, Should've" "You're Losing Me"

The Tortured Poets
Department	

    "Fortnight" "The Tortured Poets Department "My Boy Only Breaks His Favorite Toys" "Down Bad" "So Long, London" "But Daddy I Love Him" "Fresh Out the Slammer" "Florida!!!" "Guilty as Sin?" "Who's Afraid of Little Old Me?" "Loml" "I Can Do It with a Broken Heart" "The Smallest Man Who Ever Lived" "The Alchemy" "Clara Bow" "The Black Dog" "So High School" "Thank You Aimee"

Soundtrack songs	

    "Crazier" "Today Was a Fairytale" "Safe & Sound" "Eyes Open" "Sweeter than Fiction" "I Don't Wanna Live Forever" "Beautiful Ghosts" "Only the Young" "Carolina"

Featured songs	

    "Two Is Better Than One" "Half of My Heart" "Both of Us" "Babe" "Gasoline" "Renegade" "The Joker and the Queen" "The Alcott" "Us"

Other songs	

    "Best Days of Your Life" "Ronan" "Highway Don't Care" "Better Man" "Christmas Tree Farm" "All of the Girls You Loved Before"

    Category

    vte

I Prevail

    vte

MTV Video Music Award for Best Female Video

    vte

MTV Video Music Award for Best Pop Video
Authority control databases Edit this at Wikidata	

    MusicBrainz release group

Categories:

    2014 songs2014 singlesBillboard Hot 100 number-one singlesI Prevail songsCanadian Hot 100 number-one singlesMusic videos directed by Joseph KahnNumber-one singles in IcelandNumber-one singles in ScotlandSatirical songsSong recordings produced by Max MartinSong recordings produced by Shellback (record producer)Song recordings produced by Taylor SwiftSong recordings produced by Chris RoweSongs written by Taylor SwiftSongs written by Max MartinSongs written by Shellback (record producer)Taylor Swift songsSouth African Airplay Chart number-one singlesMTV Video Music Award for Best Female VideoRyan Adams songsBig Machine Records singlesElectropop songs

    This page was last edited on 9 January 2025, at 19:04 (UTC).
    Text is available under the Creative Commons Attribution-ShareAlike 4.0 License; additional terms may apply. By using this site, you agree to the Terms of Use and Privacy Policy. Wikipedia® is a registered trademark of the Wikimedia Foundation, Inc., a non-profit organization.

    Privacy policy
    About Wikipedia
    Disclaimers
    Contact Wikipedia
    Code of Conduct
    Developers
    Statistics
    Cookie statement
    Mobile view

    Wikimedia Foundation
    Powered by MediaWiki



Wikipedia The Free Encyclopedia

    Donate
    Create account
    Log in

Contents
(Top)
Background and writing

Production and release

Music and lyrics

Critical reception

Commercial performance

Music video

Awards and nominations

Live performances and other uses

Personnel

Charts

Certifications

Release history

"Love Story (Taylor's Version)"

See also

References

    External links

Love Story (Taylor Swift song)

    Article
    Talk

    Read
    Edit
    View history

Tools

Appearance
Text

    Small
    Standard
    Large

Width

    Standard
    Wide

Color (beta)

    Automatic
    Light
    Dark

Featured article
Listen to this article
From Wikipedia, the free encyclopedia
"Love Story"
Cover art of "Love Story", showing Taylor Swift in blonde braided hair and a white corset
Single by Taylor Swift
from the album Fearless
Released	September 15, 2008
Recorded	March 2008
Studio	Blackbird (Nashville)
Genre	Country pop
Length	3:55
Label	Big Machine
Songwriter(s)	Taylor Swift
Producer(s)	

    Taylor Swift Nathan Chapman

Taylor Swift singles chronology
"Should've Said No"
(2008) 	"Love Story"
(2008) 	"White Horse"
(2008)
Music video
"Love Story" on YouTube

"Love Story" is a song by the American singer-songwriter Taylor Swift. It was released as the lead single from her second studio album, Fearless, on September 15, 2008, by Big Machine Records. Inspired by a boy who was unpopular with her family and friends, Swift wrote the song using William Shakespeare's tragedy Romeo and Juliet as a reference point. The lyrics narrate a troubled romance that ends with a marriage proposal, contrary to Shakespeare's tragic conclusion. Produced by Swift and Nathan Chapman, the midtempo country pop song includes a key change after the bridge and uses acoustic instruments including banjo, fiddle, mandolin, and guitar.

At the time of the song's release, music critics praised the production but deemed the literary references ineffective. In retrospect, critics have considered it one of Swift's best singles. "Love Story" peaked atop the chart in Australia, where it was certified fourteen-times platinum by the Australian Recording Industry Association (ARIA), and reached the top five on charts in Canada, Ireland, Japan, New Zealand, and the United Kingdom. In the United States, the single peaked at number four on the Billboard Hot 100 and was the first country song to reach number one on Pop Songs. The Recording Industry Association of America (RIAA) certified it eight-times platinum. "Love Story" has sold over six million copies in the United States and 18 million copies worldwide.

Trey Fanjoy directed the accompanying music video, which stars Swift and Justin Gaston as lovers in a prior era. Drawing from historical periods such as the Renaissance and the Regency era, it won Video of the Year at both the Country Music Association Awards and CMT Music Awards in 2009. The song became a staple in Swift's live concerts and has been a part of the set lists in all of her headlining tours from the Fearless Tour (2009–2010) to the Eras Tour (2023–2024). Following a 2019 dispute regarding the ownership of Swift's back catalog, she re-recorded the song and released it as "Love Story (Taylor's Version)" in February 2021. The track topped the Hot Country Songs chart and made Swift the second artist after Dolly Parton to top that chart with both the original and re-recorded versions of a song.
Background and writing
A painting of Romeo and Juliet kissing on the balcony
Swift used Shakespeare's Romeo and Juliet as a reference point for "Love Story"; the balcony scene (pictured) is referenced in the song's opening lines.[1]

Taylor Swift moved from Pennsylvania to Nashville, Tennessee, in 2004 to pursue a career as a country singer-songwriter,[2] and in 2006, she released her first album Taylor Swift at 16 years old.[3] The album spent more weeks on the US Billboard 200 chart than any other album that was released in the 2000s decade.[4] Taylor Swift's third single "Our Song" made Swift the youngest person to single-handedly write and sing a Hot Country Songs number-one single.[5] Her success was rare for a female teenage artist; the 2000s country-music market had been dominated by adult male musicians.[6][7]

While promoting her debut album on tour in 2007 and 2008, Swift wrote songs for her second studio album Fearless.[8] She developed "Love Story" late into the production of Fearless.[9] Answering fan questions on Time in April 2009, Swift said the song was inspired by a boy whom she never dated and was one of the most romantic pieces she had written.[10] Swift recalled the reactions she received after introducing him to her family and friends: "[They] all said they didn't like him. All of them!"[11][12] This made Swift relate to the narrative of William Shakespeare's 16th-century play Romeo and Juliet, which she described as a "situation where the only people who wanted them to be together were them".[11] Reflecting on the event, Swift thought, "This is difficult but it's real, it matters"; she developed the second refrain and later the whole song around that line.[13]

Although inspired by Romeo and Juliet, Swift felt the play could have been "the best love story ever told" had it not been for Shakespeare's tragic ending in which the two characters die.[14] She thus made the narrative of "Love Story" conclude with a marriage proposal, which she deemed a happy ending the characters deserved.[14][15] Swift wrote "Love Story" on her bedroom floor in approximately 20 minutes, feeling too inspired to put the song down unfinished.[11] According to Swift, the song represents her optimistic outlook on love, which is inspired by her childhood fascination with fairy tales.[15] Looking back on "Love Story" after she released her seventh studio album Lover (2019), which is about her first experience of "love that was very real", Swift said the track is "stuff I saw on a movie [and] stuff I read mixed in with some like crush stuff that had happened in my life".[16]
Production and release

After finishing writing, Swift recorded a rough demo of "Love Story" within 15 minutes the next day.[10] She recorded the song's album version in March 2008 with the producer Nathan Chapman at Blackbird Studio in Nashville.[17] For her vocals, Chapman tried different microphones until Swift came across an Avantone CV-12 multi-pattern tube microphone that was built by the country-music artist Ray Kennedy, with whom she worked on Taylor Swift. After growing fond of the Avantone CV-12 upon testing her vocals, Swift used it to record "Love Story" and other songs. She sang the song live backed by her band, who were playing acoustic guitar, bass guitar, and drums. Chapman played other instruments, including nine acoustic guitars, and he overdubbed them on the track; he also recorded background vocals.[17] The engineer Chad Carlson recorded the track using Pro Tools and Justin Niebank mixed it using a Solid State Logic 9080 K series console and Genelec 1032 studio monitors.[17] Drew Bollman and Richard Edgeler assisted in the mixing process.[18]

"Love Story", along with the rest of Fearless, was mastered by Hank Williams at MasterMix Studios in Nashville.[18] The track uses country-music instruments such as banjo and fiddle. Big Machine Records released it to US country radio as Fearless's lead single on September 15, 2008.[17][19] Chapman mixed another version of "Love Story" for pop radio; he edited Niebank's mix using Apple Logic and muted the acoustic instruments such as banjo and fiddle.[17] The pop-radio version has an opening beat that was generated using Apple Logic's Ultrabeat, and the electric guitars were created with Amplitube Stomp I/O.[17] Rolling Stone's Keith Harris described the electric guitars as "suitably gargantuan" and louder than those on the country-radio version.[20] Big Machine in partnership with Republic Records released "Love Story" to US pop radio on October 14, 2008.[21] In the United Kingdom, "Love Story" was released on March 2, 2009, a week prior to the release of Fearless. Music Week reported that this edition was "remixed for European ears".[22]
Music and lyrics
"Love Story"
Duration: 23 seconds.0:23
A sample of the song's bridge and a key change to the final refrain. Based on William Shakespeare's Romeo and Juliet, the song's narrative ends with a marriage proposal, replacing the original ending.
Problems playing this file? See media help.

"Love Story" is a midtempo country pop song[23][24] that is driven by acoustic instruments including banjo, fiddle, mandolin, and guitar.[25] Jon Bream from the Star Tribune described the single as "pure pop with a minimalist vibe" that suits both country and pop radio.[23] According to The New York Times, despite the banjo and fiddle, the song could "easily be an emo rocker".[26] Swift's vocals have a slight twang.[27] The mix and master, according to Billboard's Kristen He, are loud and "dynamically flat ... [and are] designed to burst out of FM radio speakers".[25]

The lyrics of "Love Story" narrate a troubled romance between two characters, drawing from the lead characters in Shakespeare's Romeo and Juliet.[28] According to the psychologist Katie Barclay, the song explores feelings of love in the contexts of pain and joy.[29] "Love Story", save for the final refrain, is narrated from Juliet's perspective.[1][30] In the verses, Juliet tells the story of hers and Romeo's challenged courtship, of which her father disapproves.[31] The first verse introduces Juliet in a scene, "We were both young when I first saw you / I close my eyes and the flashback starts, I'm standing there / On a balcony in summer air", which references the balcony scene in Act II, scene ii of Shakespeare's play.[1] In the refrains, which alter slightly as the song progresses to accompany the narrative, Juliet pleads for her love interest to appear, "Romeo, take me somewhere we can be alone / I'll be waiting / All there's left to do is run."[29][30]

In the second verse, Juliet meets Romeo again in a garden and learns he must leave town because of her father's disapproval.[10] Their relationship encounters difficulties, "'Cause you were Romeo, I was a scarlet letter", referencing Nathaniel Hawthorne's The Scarlet Letter (1850).[32] According to the media-and-film scholar Iris H. Tuan, Hawthrone's "scarlet letter" imagery represents the female protagonist Hester Prynne's sin and adultery, whereas Swift's use symbolizes the forbidden love between Romeo and Juliet.[32] Juliet pleads, "This love is difficult, but it's real", which Swift said was her favorite lyric in the song.[33]

After the bridge, with accelerated drums and the harmonization of melody and vocals, the final refrain incorporates a key change up a whole step.[34] The final refrain is narrated from Romeo's perspective and tells of his marriage proposal to Juliet after he has sought her father's approval, "I talked to your dad, go pick out a white dress."[35] Whereas Shakespeare's Romeo and Juliet are secretly married without their parents' approval and both commit suicide, the characters in "Love Story" depart from that ending.[36] According to Tuan, by projecting her feelings and fantasy on a Romeo and Juliet-inspired narrative, Swift created a song that strongly resonates with an audience of teenage girls and young women.[37] Deborah Evans Price of Billboard agreed but also said "one doesn't have to be a lovestruck teen" to enjoy the song's emotional engagement.[38]
Critical reception

Blender included "Love Story" at number 73 on its 2008 year-end list,[39] and The Village Voice's Pazz & Jop critics' poll placed it at number 48.[40] In Fearless reviews, many critics complimented the production; Sean Daly from the St. Petersburg Times,[41] Rob Sheffield from Blender[42] and Stephen Thomas Erlewine from AllMusic selected the track as an album highlight.[43] Deborah Evans Price of Billboard praised the "swirling, dreamy" production and said Swift's success in the country-music market "could only gain momentum".[38] Others including The Boston Globe's James Reed[30] and USA Today's Elysa Gardner deemed "Love Story" an example of Swift's songwriting abilities at a young age; the latter appreciated the song for earnestly portraying teenage feelings "rather than [being] a mouthpiece for a bunch of older pros' collective notion of adolescent yearning".[44]

Some critics were more reserved in their praise and took issue with the literary references. In a four-stars-out-of-five rating of the song for the BBC, Fraser McAlpine deemed the Shakespearean reference not as sophisticated as its premise and the lyrics generic, but he praised the production and wrote, "It's great to see a big pop song being used as a method of direct story telling."[28] The musicologist James E. Perone commented: "the melodic hooks are strong enough to overcome the predictability of the lyrics."[45] Jon Bream from the Star Tribune deemed the single inferior to Swift's debut country-music single "Tim McGraw" (2006) but commended the production as catchy.[23] In a Slant Magazine review, Jonathan Keefe was impressed by Swift's melodic songwriting for creating "massive pop hooks" but found the references to Romeo and Juliet "point-missing" and The Scarlet Letter "inexplicable". Keefe deemed the lyrics lacking in creativity and disapproved of Swift's "clipped phrasing" in the refrain.[46]

In retrospective commentaries, the English-language professor Robert N. Watson and the music journalist Annie Zaleski deemed "Love Story" a memorable song thanks to its Shakespearean narrative; the former regarded it as an evidence of Swift's status as "the twenty-first-century's most popular songwriter of failed love affairs"[47] and the latter wrote that the song deserves to be "a love story for the ages".[48] "Love Story" was included on best-of lists including Taste of Country's "Top 100 Country Songs" (2016),[49] Time Out's "35 Best Country Songs of All Time" (2022),[50] and Billboard's "Top 50 Country Love Songs of All Time" (2022).[24]

Critics have rated "Love Story" high in rankings of Swift's songs; these include Hannah Mylrea from NME (2020), who ranked it fifth out of 160 songs,[51] Jane Song from Paste (2020), 13th out of 158,[52] and Nate Jones from Vulture (2024), 11th out of 245.[53] In another ranking of Swift's select 100 tracks for The Independent, Roisin O'Connor placed "Love Story" at number 15 and said it showcases Swift as a songwriter who "understands the power of a forbidden romance".[54] Alexis Petridis from The Guardian placed it second, behind "Blank Space" (2014), on his 2019 ranking of Swift's 44 singles. He said of the literary references: "[If] the references to Shakespeare and Hawthorn seem clumsy, they are clumsy in a believably teenage way."[55]
Commercial performance

In the United States, "Love Story" debuted at number 16 on the Billboard Hot 100 and at number 25 on the Hot Country Songs chart, both dated September 27, 2008.[56][57] The next week, it reached number five on the Hot 100.[58] The single peaked at number four on the Hot 100 chart dated January 17, 2009, and spent 49 weeks on the chart.[59] It spent two weeks atop the Hot Country Songs chart.[60] On the Pop Songs chart, which tracks US pop radio, "Love Story" reached number one on the week ending February 28, 2009.[61] It became the first song to top both the country-radio and pop-radio charts and surpassed the number-three-peaking "You're Still the One" (1998) by Shania Twain as the highest-charting country crossover to pop radio.[62]

On other Billboard airplay charts, "Love Story" peaked at number one on Adult Contemporary and number three on Adult Pop Songs.[63][64] Together with "Teardrops on My Guitar" (2007), "Love Story" made Swift the first artist in the 2000s decade to have two titles each reach the top 10 of four airplay charts; Hot Country Songs, Pop Songs, Adult Pop Songs, and Adult Contemporary.[65] It topped the 2009 year-end Radio Songs chart.[66] By February 2009, it was the first country song to sell three million downloads.[67] In 2015, the Recording Industry Association of America (RIAA) certified "Love Story" eight-times platinum.[68] The single had sold 6.2 million copies in the United States by October 2022 and became Swift's highest-selling single in the nation.[69]

"Love Story" was Swift's first number-one single in Australia,[70] where it was certified fourteen-times platinum.[71] It peaked within the top five of singles charts in Japan (three),[72] and the wider English-speaking world: the United Kingdom (two),[73] Ireland (three),[74] New Zealand (three),[75] Canada (four),[76] and Scotland (five).[77] In mainland Europe, the single peaked at number ten on the European Hot 100 Singles chart,[78] number four in the Czech Republic,[79] number six in Hungary,[80] number seven in Norway,[81] and number ten in Sweden.[82] "Love Story" was certified triple platinum in the United Kingdom and New Zealand,[83][84] double platinum in Canada,[85] platinum in Denmark and Germany,[86][87] and gold in Italy and Japan.[88] It sold 6.5 million digital copies worldwide and was the sixth-most-downloaded single of 2009.[89] By February 2021, estimated worldwide sales of "Love Story" stood at 18 million units.[90]
Music video
Development and filming

Trey Fanjoy, who had worked with Swift on previous music videos, directed "Love Story".[91] Swift was inspired by historical eras such as the Middle Ages, the Renaissance, and the Regency to make a period-piece-styled video with a timeless narrative that "could happen in the 1700s, 1800s, or 2008".[91] She spent six months searching for the male lead and upon recommendation from an acquaintance chose Justin Gaston, a fashion model who was competing in the television series Nashville Star.[92][93] After Gaston was eliminated from the show, Swift contacted him to appear in the video.[91] She believed Gaston was a perfect choice for the male lead: "I was so impressed by the way his [expressions] were in the video. Without even saying anything, he would just do a certain glance and it really came across well."[92]

The music video was filmed within two days in August 2008 in Tennessee. The crew considered traveling to Europe to find a castle for the video's setting but settled on Castle Gwynn in Arrington; the castle was built in 1973 and is part of the annual Tennessee Renaissance Festival.[91] Wardrobe for the video—except Swift's dress for the balcony scene, which was designed by Sandi Spika with inspiration and suggestions from Swift—was supplied by Jacquard Fabrics.[91] On the first day, the balcony and field scenes were filmed. The second day's filming included the ballroom scene was filmed with 20 dancers from Cumberland University in Lebanon; Swift learned the choreography 15 minutes prior to filming.[91] She invited some fans who were university students from other states to fly to Nashville and film the video with her.[94] "Love Story" premiered on September 12, 2008, on CMT.[95] Behind-the-scenes footage of the music video's production was aired on Great American Country on November 12, 2008.[96]
Synopsis and commentary

The video starts with Swift wearing a black sweater and jeans; she walks through a college campus and sees Gaston reading under a tree. As they make eye contact, the video transitions to a balcony, on which Swift is wearing a corset and gown. The video switches to a ballroom where Gaston and Swift dance together, after which Gaston whispers into Swift's ear. Swift is next shown walking into a garden with a lantern at night. She meets with Gaston and they have a date before parting ways. Later, Swift again stands on the balcony looking out from the window. She sees Gaston running across a field towards her and she immediately runs down the staircase to meet him. The video then switches back to the modern-day college campus, where Gaston walks toward Swift and they gaze into each other's eyes, and the video ends.[91]

Spin wrote that the video appears to have been filmed on an "HBO-looking budget" with "elaborate, pseudo-medieval set pieces"; according to the magazine, rather than alluding to Shakespeare's Romeo and Juliet, the narrative resembles "Rapunzel", especially the part in which Swift's character waits for her lover atop a castle.[97] According to Glamour, Swift's fashion in the video reinforces the lyrical theme, "[She] literally wore a medieval ball gown while playing the Juliet to an actor's Romeo."[98] In a 2010 Billboard interview, Swift reflected on the video's fairy-tale-inspired wedding setting: "I'm not really that girl who dreams about her wedding day. It just seems like the idealistic, happy-ever-after [moment]."[99]
Awards and nominations

"Love Story" won Song of the Year at the Country Awards in 2009 and Pop Awards in 2010, both of which were held by Broadcast Music, Inc. (BMI) to honor the year's most-performed songs on US radio and television.[100] It marked Swift's second consecutive Song of the Year win at the BMI Country Awards, following "Teardrops on My Guitar" in 2008.[101] Swift, who was 20, was the youngest songwriter to win Song of the Year at the BMI Pop Awards.[102] At the Australian APRA Awards, "Love Story" was nominated for International Work of the Year.[103]

It received nominations at the People's Choice Awards (Favorite Country Song, which went to Carrie Underwood's "Last Name"),[104] Nickelodeon Australian Kids' Choice Awards (Favorite Song, which went to the Black Eyed Peas' "Boom Boom Pow"),[105] and Teen Choice Awards (Choice Love Song, which went to David Archuleta's "Crush").[106][107] The music video was nominated for Video of the Year at the 45th Academy of Country Music Awards, but it lost to Brad Paisley's "Waitin' on a Woman" (2008).[108][109] At the 2009 CMT Music Awards, it won Video of the Year and Female Video of the Year.[110] It also won Music Video of the Year at the 43rd Country Music Association Awards[111] and Favorite International Video at the Philippine Myx Music Awards 2010.[112]
Live performances and other uses
Taylor Swift singing on a flying balcony on the Speak Now tour
Swift performing "Love Story" on a flying balcony at the Speak Now World Tour in 2011

"Love Story" has become a staple in Swift's concerts—as of July 2023, she had performed the song live over 500 times.[113] The song became a number in Swift's shows during which many couples get engaged, specifically after the bridge that has lyrics about Romeo proposing to Juliet.[114]

During promotion of Fearless in 2008 and 2009, Swift performed "Love Story" on television shows including Good Morning America, Late Show with David Letterman, The Today Show,[115] Dancing with the Stars,[116] The Ellen DeGeneres Show,[117] and Saturday Night Live.[118] At the 2008 Country Music Association Awards, she re-enacted the music video for "Love Story", performing the song on a ballroom stage-setting with Gaston playing the love interest.[119] Swift and the English band Def Leppard performed "Love Story", among other tracks from each artist's repertoire, for a CMT Crossroads episode that was recorded in October 2008; the performance was released on DVD in 2009.[120] In the United Kingdom, Swift sang "Love Story" on the BBC charity telethon Children in Need, to which she donated £13,000 afterward.[116]

"Love Story" was part of the set lists for many of Swift's 2009 headline festival performances, including Houston Livestock Show and Rodeo,[121] Florida Strawberry Festival,[122] Sound Relief,[123] the CMA Music Festival,[124] and Craven Country Jamboree.[125] She included the song in the set list of her first headlining concert tour the Fearless Tour (2009–2010). The song's performances began with backup dancers dressed in Victorian clothing, dancing to Pachelbel's Canon as a castle backdrop was projected onto the stage.[126] Swift emerged from below to an upper level of the stage; she wore an 18th-century-styled crimson gown with golden accents.[127] For the final refrain, Swift hid behind backup dancers as she changed into a white wedding dress and a jeweled headband.[128][129] The live performances of "Love Story" were recorded and released on the DVD Journey to Fearless in 2011.[130]
Taylor Swift on the 1989 tour
Swift singing a synth-pop version of "Love Story" on the 1989 World Tour in 2015

"Love Story" was the final song on the set list of Swift's second headlining tour, the Speak Now World Tour (2011–2012).[131] Swift wore a white sundress and sang the song while roaming across the stage on a flying balcony as confetti rained down and fireworks exploded on stage.[132] The song was part of Swift's performance at BBC Radio 1's Teen Awards in October 2012; she appeared in a white dress before changing into silver hot pants and a sheer black top.[133] Swift sang the song later the same month as part of a VH1 Storytellers episode that was recorded at Harvey Mudd College in California.[134] On January 25, 2013, Swift performed an acoustic version of "Love Story" at the Los Premios 40 Principales in Spain.[135] She again included the song in the set list of her third headlining tour the Red Tour (2013–2014), in which she sang it while wearing a white gown.[136]

At the 2014 iHeartRadio Music Awards, Swift performed an arena rock version of "Love Story".[137] During concerts of her fourth headlining tour the 1989 World Tour (2015), she rearranged the song as a synth-pop ballad and sang it while standing on an elevated platform that whisked around the venue.[138][139] Commenting on the 1989 World Tour rearrangement, Jane Song from Paste said "Love Story" "will continue to be one of [Swift's] calling cards".[52] Swift again included "Love Story" in the set list of her fifth concert tour, 2018's Reputation Stadium Tour, in which she performed it as part of a medley with her singles "Style" and "You Belong with Me".[140]

On April 23, 2019, she performed a piano rendition of "Love Story" at Lincoln Center for the Performing Arts during the Time 100 Gala, in which she was honored as one of the year's "most influential people".[141] On September 9, Swift performed the song at the City of Lover one-off concert in Paris.[142] At the American Music Awards of 2019, at which she was awarded "Artist of the Decade", Swift performed "Love Story" as part of a medley with "The Man", "I Knew You Were Trouble", "Blank Space", and "Shake It Off".[143] On July 21, 2022, at a concert of Haim's One More Haim Tour in London, Swift made a guest appearance and performed "Love Story" as part of a mashup with "Gasoline".[144] She again included "Love Story" in the regular set list of her sixth headlining concert tour, the Eras Tour (2023–2024).[145]

"Love Story" has been parodied and adapted into popular-culture events. For the 2009 CMT Music Awards, Swift and the rapper T-Pain recorded a parody titled "Thug Story", in which they rap and sing with Auto-Tune; the parody aired as part of the awards ceremony's cold open.[146] In August 2020, an unofficial house remix of "Love Story" by the American DJ Disco Lines went viral on the video-sharing platform TikTok.[147] The Disco Lines remix charted at number 37 on Poland's airplay chart in October 2020.[148] Following the cancellation of Swift's three Eras Tour shows at Ernst-Happel-Stadion in Vienna due to a terror plot, the British rock band Coldplay and the American singer Maggie Rogers covered "Love Story" on the Music of the Spheres World Tour at the same venues weeks later as a tribute.[149]
Personnel

    Taylor Swift – lead vocals, songwriter, producer, backing vocals
    Nathan Chapman – producer, backing vocals
    Drew Bollman – assistant mixer
    Chad Carslon – recording engineer
    Richard Edgeler – assistant recording engineer, assistant mixer
    Justin Niebank – mixer
    Tim Van der Kull – additional guitar
    Jeremy "Jim Bob" Wheatley – additional recording engineer, additional mixer
    Caitlin Evanson – backing vocals

Charts
Weekly charts
2008–2009 weekly chart performance for "Love Story" Chart (2008–2009) 	Peak
position
Australia (ARIA)[70] 	1
Austria (Ö3 Austria Top 40)[150] 	30
Belgium (Ultratip Bubbling Under Flanders)[151] 	4
Belgium (Ultratop 50 Wallonia)[152] 	39
Canada (Canadian Hot 100)[76] 	4
Canada AC (Billboard)[153] 	1
Canada CHR/Top 40 (Billboard)[154] 	4
Canada Country (Billboard)[155] 	1
Canada Hot AC (Billboard)[156] 	3
CIS Airplay (TopHit)[157] 	180
Czech Republic (Rádio Top 100 Oficiální)[79] 	4
Denmark (Tracklisten)[158] 	16
European Hot 100 Singles (Billboard)[78] 	10
Euro Digital Song Sales (Billboard)[159] 	5
Finland Download (Latauslista)[160] 	17
France (SNEP)[161] 	14
Germany (GfK)[162] 	22
Hungary (Single Top 40)[80] 	6
Ireland (IRMA)[74] 	3
Japan (Japan Hot 100)[72] 	3
Japan Adult Contemporary (Billboard)[163] 	1
Mexico Ingles Airplay (Billboard)[164] 	6
Netherlands (Single Top 100)[165] 	13
New Zealand (Recorded Music NZ)[75] 	3
Norway (VG-lista)[81] 	7
Scotland (OCC)[77] 	5
Slovakia (Rádio Top 100)[166] 	13
Spain (PROMUSICAE)[167] 	47
Sweden (Sverigetopplistan)[82] 	10
Switzerland (Schweizer Hitparade)[168] 	50
UK Singles (OCC)[73] 	2
US Billboard Hot 100[59] 	4
US Adult Contemporary (Billboard)[63] 	1
US Adult Pop Airplay (Billboard)[64] 	3
US Hot Country Songs (Billboard)[60] 	1
US Latin Pop Airplay (Billboard)[169] 	35
US Pop Airplay (Billboard)[61] 	1
US Pop 100 (Billboard)[170] 	3
2024 weekly chart performance for "Love Story" Chart (2024) 	Peak
position
Malaysia International (RIM)[171] 	12
Portugal (AFP)[172] 	77
Singapore (RIAS)[173] 	2
	
Year-end charts
2008 year-end charts for "Love Story" Chart (2008) 	Position
US Billboard Hot 100[174] 	81
US Hot Country Songs (Billboard)[175] 	55
2009 year-end charts for "Love Story" Chart (2009) 	Position
Australia (ARIA)[176] 	3
Canada (Canadian Hot 100)[177] 	8
European Hot 100 Singles (Billboard)[178] 	94
France (SNEP)[179] 	94
Japan (Japan Hot 100)[180] 	54
New Zealand (RMNZ)[181] 	13
Sweden (Sverigetopplistan)[182] 	47
UK Singles (Official Charts Company)[183] 	29
US Billboard Hot 100[184] 	5
US Adult Contemporary (Billboard)[185] 	2
US Adult Pop Songs (Billboard)[186] 	11
US Pop Songs (Billboard)[187] 	8
Decade-end charts
2000–2009 decade-end charts for "Love Story" Chart (2000–2009) 	Position
Australia (ARIA)[188] 	10
US Billboard Hot 100[189] 	73

Certifications
Certifications for "Love Story" Region 	Certification 	Certified units/sales
Australia (ARIA)[71] 	14× Platinum 	980,000‡
Austria (IFPI Austria)[190] 	Platinum 	30,000*
Brazil (Pro-Música Brasil)[191] 	2× Platinum 	120,000‡
Canada (Music Canada)[85] 	2× Platinum 	160,000*
Denmark (IFPI Danmark)[86] 	Platinum 	90,000‡
Germany (BVMI)[87] 	Platinum 	300,000‡
Italy (FIMI)[192] 	Gold 	50,000‡
Japan (RIAJ)[88] 	Gold 	100,000*
New Zealand (RMNZ)[84] 	3× Platinum 	90,000‡
Spain (PROMUSICAE)[193] 	Gold 	30,000‡
United Kingdom (BPI)[83] 	3× Platinum 	1,800,000‡
United States (RIAA)[68] 	8× Platinum 	8,000,000‡
United States (RIAA)[194]
Mastertone 	Platinum 	1,000,000*

* Sales figures based on certification alone.
‡ Sales+streaming figures based on certification alone.
Release history
List of release dates and formats for "Love Story" Region 	Release date 	Format 	Version 	Label 	Ref.
United States 	September 15, 2008 	Country radio 	Original 	Big Machine 	[19]
October 14, 2008 	Contemporary hit radio 	

    Big MachineRepublic

	[21]
Various 	January 6, 2009 	Digital download 	Pop Mix 	Big Machine 	[195]
February 6, 2009 	Stripped 	[196]
February 27, 2009 	Digital Dog Radio Mix 	[197]
United Kingdom 	March 2, 2009 	Original 	

    MercuryUniversal Music

	[198][199]
Germany 	April 24, 2009 	CD single 	Universal Music 	[200]
"Love Story (Taylor's Version)"
"Love Story (Taylor's Version)"
Cover artwork of "Love Story (Taylor's Version)" featuring Taylor Swift in a white dress
Promotional single by Taylor Swift
from the album Fearless (Taylor's Version)
Released	February 12, 2021
Studio	Blackbird (Nashville)
Genre	Country pop
Length	3:56
Label	Republic
Songwriter(s)	Taylor Swift
Producer(s)	

    Taylor Swift Christopher Rowe

Lyric video
"Love Story" (Taylor's Version) on YouTube

After signing a new contract with Republic Records, Swift began re-recording her first six studio albums, including Fearless, in November 2020.[201] The decision came after a 2019 public dispute between Swift and the talent manager Scooter Braun, who acquired Big Machine Records, including the masters of Swift's albums the label had released.[202][203] By re-recording her catalog, Swift had full ownership of the new masters, including the copyright licensing of her songs, devaluing the Big Machine-owned masters.[204]

Swift re-recorded "Love Story" and titled it "Love Story (Taylor's Version)". An excerpt of the re-recording was used in a Match.com advertisement in December 2020.[205] "Love Story (Taylor's Version)" was the first re-recorded track she released;[206] it was made available for download and streaming on February 12, 2021, preceding the release of the re-recorded album Fearless (Taylor's Version) in April.[207][208] An EDM version of "Love Story (Taylor's Version)" remixed by Swedish producer Elvira was released on March 26, 2021, and was included on the deluxe edition of Fearless (Taylor's Version).[209]
Production

"Love Story (Taylor's Version)" was produced by Swift and the Nashville-based producer Christopher Rowe. It was recorded by David Payne at Blackbird Studio, with additional recording by Rowe at Prime Recording and Studio 13, all of which are in Nashville. Sam Holland recorded Swift's vocals at Conway Recording Studios in Los Angeles; Serban Ghenea mixed the re-recording at MixStar Studios in Virginia Beach, Virginia; and Randy Merrill mastered it at Sterling Sound in Edgewater, New Jersey.[210] Swift invited some of the musicians who worked on the 2008 version to re-record with her; these participants include Jonathan Yudkin on fiddle, Amos Heller on bass guitar, and Caitlin Evanson on harmony vocals; they were part of Swift's touring band and had played "Love Story" with her many times.[210]

According to critics, the production of "Love Story (Taylor's Version)" is faithful to that of the 2008 version.[210][211] They noticed changes in the timbre of Swift's vocals, which have a fuller tone and an absence of the country-music twang;[27][212] The Atlantic's Shirley Li found Swift's voice "much richer" with a controlled tone and precise staccato.[213] Swift said re-recording "Love Story" made her realize how she had improved as a singer and how her "voice was so teenaged" in the old recordings.[214]

The re-recording's instruments are sharper and more distinct, with clearer sounds of the banjo, cymbals, and fiddle; stronger drums; a more-clearly defined bass; less-harsh electric guitars; and lowered harmonies in the mix.[25][212][215] In Billboard, Kristen He said whereas the instruments on the 2008 version blended into a "wall of sound", the production of "Love Story (Taylor's Version)" highlighted individual instruments.[25]
Reception

In reviews, critics praised "Love Story (Taylor's Version)" for being faithful to the original version and felt it was improved upon with polished production and Swift's mature vocals.[212][213][216] A few welcomed the re-recording as Swift's display of ownership of her music.[211][215] Reviews from Rolling Stone's Simon Vozick-Levinson and Los Angeles Times's Mikael Wood dubbed the re-recording an update of a "classic" song about teenage sentiments.[211][217] Mark Savage from BBC News said Swift's improved vocals retain the teenage feelings,[212] but The Atlantic's Shirley Li and NME's Hannah Mylrea said they were more powerful, which introduces a sense of wistfulness and therefore loses the earnestness of the 2008 version.[213][216] According to Robert Christgau, "Swift's voice retains a great deal of freshness" but he questioned the value of her re-recording of early songs, saying he did not comprehend how he would pay for the re-recordings.[218]

In the United States, "Love Story (Taylor's Version)" debuted atop the Hot Country Songs chart, giving Swift her eighth number-one single and first number-one debut. With this achievement, she became the first artist to lead the chart in the 2000s, 2010s, and 2020s, and the second artist to have a number one with both the original and re-recorded version of a song, after Dolly Parton with "I Will Always Love You". On other Billboard charts, "Love Story (Taylor's Version)" topped Digital Song Sales (Swift's record-extending 22nd number one), Country Digital Song Sales (record-extending 15th number one), and Country Streaming Songs. The song debuted and peaked at number 11 on the Billboard Hot 100, her record-extending 129th chart entry.[219] The re-recording peaked at number seven on the Billboard Global 200.[57] It topped the singles chart in Malaysia[220] and reached the top 10 in Canada,[76] Ireland,[221] and Singapore.[222] It also charted at number 12 in the United Kingdom,[223] where it was certified gold,[224] and number 18 in New Zealand.[225]

In October 2021, Billboard reported radio stations in the United States played "Love Story (Taylor's Version)" and other re-recordings infrequently compared to the originals; reasons given were that the re-recordings were insufficiently distinctive, that they had less audience demand for Swift's older songs than her newer ones, and they were difficult to categorize in radio format terms, as well as there being no financial incentive from Swift to promote the re-recordings to radio as radio stations do not have to pay the owners of the master recording every time they play a song and Swift would still receive songwriting royalties no matter what version was played.[226] At the 2022 CMT Music Awards, the re-recording won the inaugural Trending Comeback Song of the Year; CMT created the category to honor "iconic stars and their hits that not only stood the test of time but also recently found new popularity".[227]
Credits and personnel

    Taylor Swift – lead vocals, songwriting, production
    Christopher Rowe – production, record engineering
    David Payne – record engineering
    John Hanes – engineering
    Randy Merrill – master engineering
    Serban Ghenea – mixing
    Sam Holland – vocal engineering
    Sean Badum – assistant recording engineering
    Mike Meadows – backing vocals, acoustic guitar, banjo, mandolin
    Paul Sidoti – backing vocals, electric guitar
    Caitlin Evanson – backing vocals
    Amos Heller – bass
    Matt Billingslea – drums
    Max Bernstein – electric guitar
    Jonathan Yudkin – fiddle

Charts
Weekly charts
Weekly chart performance for "Love Story (Taylor's Version)" Chart (2021–2022) 	Peak
position
Australia (ARIA)[228] 	21
Belgium (Ultratip Bubbling Under Flanders)[229] 	24
Canada (Canadian Hot 100)[76] 	7
Canada AC (Billboard)[153] 	23
Euro Digital Song Sales (Billboard)[230] 	10
Global 200 (Billboard)[231] 	7
Ireland (IRMA)[221] 	7
Latvia (EHR)[232] 	2
Malaysia (RIM)[220] 	1
Netherlands (Single Tip)[233] 	6
New Zealand (Recorded Music NZ)[225] 	18
Portugal (AFP)[234] 	68
Singapore (RIAS)[222] 	3
Sweden (Sverigetopplistan)[235] 	62
UK Singles (OCC)[223] 	12
US Billboard Hot 100[59] 	11
US Adult Contemporary (Billboard)[63] 	25
US Adult Top 40 (Billboard)[64] 	39
US Country Airplay (Billboard)[236] 	57
US Hot Country Songs (Billboard)[60] 	1
US Rolling Stone Top 100[237] 	4
	
Year-end charts
Year-end chart performance for "Love Story (Taylor's Version)" Chart (2021) 	Position
US Hot Country Songs (Billboard)[238] 	80

Certifications
Certifications for "Love Story (Taylor's Version)" Region 	Certification 	Certified units/sales
Australia (ARIA)[71] 	3× Platinum 	210,000‡
Brazil (Pro-Música Brasil)[239] 	2× Platinum 	80,000‡
New Zealand (RMNZ)[240] 	Platinum 	30,000‡
Poland (ZPAV)[241] 	Gold 	25,000‡
Spain (PROMUSICAE)[242] 	Gold 	30,000‡
United Kingdom (BPI)[224] 	Platinum 	600,000‡

‡ Sales+streaming figures based on certification alone.
Release history
List of release dates and formats for "Love Story (Taylor's Version)" Region 	Date 	Format 	Version 	Label 	Ref.
Various 	February 12, 2021 	

    Digital downloadstreaming

	Original 	Republic 	[243]
March 26, 2021 	Elvira remix 	[244]
See also

    List of best-selling singles
    List of best-selling singles in Australia
    List of best-selling singles in the United States
    List of number-one singles of 2009 (Australia)
    List of Billboard Adult Contemporary number ones of 2009
    List of Hot Country Songs number ones of 2008
    List of Billboard Mainstream Top 40 number-one songs of 2009
    List of top 10 singles in 2021 (Ireland)
    List of Billboard number-one country songs of 2021

References

Tuan 2020, p. 27.
Malec, Jim (May 2, 2011). "Taylor Swift: The Garden In The Machine". American Songwriter. Archived from the original on May 10, 2012. Retrieved May 21, 2012.
Widdicombe, Lizzie (October 10, 2011). "You Belong With Me". The New Yorker. Archived from the original on July 24, 2014. Retrieved October 11, 2011.
Trust, Gary (October 29, 2009). "Chart Beat Thursday: Taylor Swift, Tim McGraw Linked Again". Billboard. Archived from the original on June 26, 2019. Retrieved July 12, 2019.
"Taylor Swift". Songwriters Hall of Fame. Archived from the original on February 12, 2021. Retrieved February 27, 2021.
Malec, Jim (May 2, 2011). "Taylor Swift: The Garden in the Machine". American Songwriter. p. 4. Archived from the original on March 26, 2016. Retrieved May 21, 2012. "It also established her as one of only a handful of new female voices to break out at country radio in a decade that was almost completely dominated by men"
Caramanica, Jon (November 9, 2008). "My Music, MySpace, My Life". The New York Times. Archived from the original on April 11, 2009. Retrieved February 28, 2010.
Tucker, Ken (March 26, 2008). "The Billboard Q&A: Taylor Swift". Billboard. Archived from the original on July 5, 2013. Retrieved June 21, 2011.
Scaggs, Austin (January 25, 2010). "Taylor's Time: Catching Up With Taylor Swift". Rolling Stone. Archived from the original on August 15, 2012. Retrieved February 1, 2010.
Spencer 2010, p. 62.
"Interview with Taylor Swift". Time. April 23, 2009. Archived from the original on May 31, 2012. Retrieved February 12, 2011.
Stahl, Lesley (November 20, 2011). Taylor Swift: A Young Star's Meteoric Rise (Television broadcast). 60 Minutes. Produced by Shari Finkelstein. CBS News. Archived from the original on November 24, 2022. Retrieved November 24, 2022.
Bells, Leigh (November 28, 2008). "Taylor Swift Responds!". Teen Vogue. Archived from the original on February 25, 2011. Retrieved February 12, 2011.
Lewis, Randy (October 26, 2008). "She's Writing Her Future". Los Angeles Times. Archived from the original on August 22, 2021. Retrieved February 12, 2011.
Roznovsky, Lindsey (November 10, 2008). "Taylor Swift's Fascination with Fairy Tales Comes Through on New Album". CMT News. Archived from the original on October 20, 2012. Retrieved February 13, 2011.
Aniftos, Rania (October 30, 2019). "Taylor Swift Compares Lover to Reputation, Talks #MeToo Movement With Zane Lowe For Beats 1 Interview". Billboard. Archived from the original on April 8, 2020. Retrieved November 14, 2019.
Walsh, Christopher (April 17, 2009). "Taylor Swift — Love Story". ProAudio Review. The Wicks Group. Archived from the original on December 19, 2010. Retrieved February 14, 2011.
Swift, Taylor (2008). Fearless (CD). Big Machine Records. BMRATS0200.
"Country Aircheck Chart Info" (PDF). Country Aircheck. No. 106. Nashville. September 8, 2008. p. 12. Archived (PDF) from the original on August 28, 2021. Retrieved August 27, 2021.
Harris, Keith (September 9, 2014). "Trace Taylor Swift's Country-to-Pop Transformation in 5 Songs". Rolling Stone. Archived from the original on December 8, 2022. Retrieved August 27, 2022.
"Available for Airplay". FMQB. Archived from the original on May 3, 2012. Retrieved May 3, 2012.
"Campaign Focus: Taylor Swift". Music Week. January 24, 2009. p. 15. ProQuest 232143764.
Bream, Jon (October 5, 2008). "Download This". Star Tribune. p. E2. ProQuest 428001558.
Dauphin, Chuck; Pascual, Danielle (July 1, 2022). "Top 50 Country Love Songs of All Time". Billboard. Archived from the original on December 9, 2022. Retrieved August 26, 2022.
He, Kristen (February 14, 2021). "Taylor Swift's 'Love Story' Re-Recording Gently Reinvents a Modern Classic". Billboard. Archived from the original on April 23, 2021. Retrieved February 14, 2021.
"C.M.A. Again Picks Chesney as Entertainer of the Year". The New York Times. November 12, 2008. ProQuest 2221241354.
Hughes, William (February 12, 2021). "Taylor Swift Just Unleashed the Full 'Taylor's Version' of 2008's 'Love Story'". The A.V. Club. Archived from the original on December 9, 2022. Retrieved August 26, 2022.
McAlpine, Fraser (February 28, 2009). "Taylor Swift – 'Love Story'". BBC. Archived from the original on November 7, 2010. Retrieved March 8, 2011.
Barclay 2018, p. 546.
Reed, James (November 10, 2008). "Young Country Star's Fearless Proves She's Just That, and More". The Boston Globe. Archived from the original on January 13, 2010. Retrieved March 8, 2011.
Perone 2017, p. 21; Barclay 2018, p. 546.
Tuan 2020, p. 28.
Barclay 2018, p. 546; Spencer 2010, p. 62.
Sloan, Harding & Gottlieb 2019, pp. 34–35; Barclay 2018, p. 547.
Sloan 2021, p. 15; Barclay 2018, p. 547; Tuan 2020, p. 28.
Perone 2017, p. 21; Tuan 2020, p. 28; Spencer 2010, p. 65.
Tuan 2020, pp. 30–31.
Price, Deborah Evans (October 11, 2008). "Singles: 'Love Story'". Billboard. Vol. 120, no. 41. p. 68. ISSN 0006-2510. Archived from the original on February 7, 2023. Retrieved March 8, 2011 – via Google Books.
"Blender's Top 33 Albums and 144 Songs of 2008". Blender. Vol. 76, no. December 2008/January 2009. November 22, 2008. p. 34.
"Pazz & Jop 2008". The Village Voice. Archived from the original on February 1, 2009. Retrieved February 1, 2009.
Daly, Sean (November 23, 2008). "Album Reviews". St. Petersburg Times. p. L4. ProQuest 264265180.
Sheffield, Rob. "Taylor Swift: Fearless". Blender. Archived from the original on December 16, 2008. Retrieved October 8, 2018.
Thomas Erlewine, Stephen. "Fearless – Taylor Swift". AllMusic. Archived from the original on October 8, 2018. Retrieved October 8, 2018.
Gardner, Elysa (November 11, 2008). "Taylor Swift Hits All the Right Words". USA Today. Archived from the original on April 28, 2016. Retrieved April 17, 2016.
Perone 2017, p. 21.
Keefe, Jonathan (November 16, 2008). "Taylor Swift: Fearless". Slant Magazine. Archived from the original on March 15, 2011. Retrieved March 8, 2011.
Watson 2015, p. 83.
Zaleski 2024, p. 34.
Pacella, Megan (June 24, 2012). "No. 17: Taylor Swift, 'Love Story' – Top 100 Country Songs". Taste of Country. Archived from the original on May 30, 2014. Retrieved May 29, 2014.
"The 35 Best Country Songs of All Time". Time Out. August 9, 2022. Archived from the original on November 29, 2022. Retrieved September 21, 2022.
Mylrea, Hannah (September 8, 2020). "Every Taylor Swift Song Ranked In Order of Greatness". NME. Archived from the original on September 8, 2020. Retrieved December 1, 2021.
Song, Jane (February 11, 2020). "All 158 Taylor Swift Songs, Ranked". Paste. Archived from the original on April 13, 2020. Retrieved December 9, 2020.
Jones, Nate (May 20, 2024). "All 245 Taylor Swift Songs, Ranked". Vulture. Archived from the original on October 7, 2024. Retrieved November 5, 2024.
O'Connor, Roisin (August 23, 2019). "Taylor Swift: Her 100 Album Tracks – Ranked". The Independent. Archived from the original on December 3, 2019. Retrieved September 14, 2019.
Petridis, Alexis (April 26, 2019). "Taylor Swift's Singles – Ranked". The Guardian. Archived from the original on April 27, 2019. Retrieved January 24, 2021.
Cohen, Jonathan (September 18, 2008). "Pink Notches First Solo Hot 100 No. 1". Billboard. Archived from the original on April 30, 2013. Retrieved March 5, 2011.
Tucker, Ken (October 25, 2008). "Taylor Swift Goes Global". Billboard. Vol. 120, no. 43. pp. 22–25. ProQuest 227230140.
Cohen, Jonathan (September 25, 2008). "T.I. Back Atop Hot 100, Kanye Debuts High". Billboard. Archived from the original on May 2, 2013. Retrieved March 5, 2011.
"Taylor Swift Chart History (Hot 100)". Billboard. Retrieved October 21, 2022.
"Taylor Swift Chart History (Hot Country Songs)". Billboard. Retrieved October 21, 2022.
"Taylor Swift Chart History (Pop Songs)". Billboard. Retrieved January 12, 2012.
Trust, Gary (December 15, 2009). "Best of 2009: Part 1". Billboard. Archived from the original on March 3, 2013. Retrieved March 5, 2011.
"Taylor Swift Chart History (Adult Contemporary)". Billboard. Retrieved October 21, 2022.
"Taylor Swift Chart History (Adult Pop Songs)". Billboard. Retrieved October 21, 2022.
Trust, Gary; Caulfield, Keith (January 31, 2009). "Fearless Feats". Billboard. Vol. 121, no. 4. p. 40. ProQuest 227278627.
"The Best of 2009: The Year in Music". Billboard. Archived from the original on January 25, 2016. Retrieved September 17, 2015.
Williams, Rob (July 10, 2009). "Crossover Queen". Winnipeg Free Press. p. D1. ProQuest 752237242.
"American single certifications – Taylor Swift – Love Story". Recording Industry Association of America. Retrieved January 30, 2022.
Trust, Gary (October 21, 2022). "Ask Billboard: Taylor Swift's Career Streaming, Airplay & Sales, Ahead of the Chart Debut of Midnights". Billboard. Archived from the original on October 21, 2022. Retrieved October 21, 2022.
"Taylor Swift – Love Story". ARIA Top 50 Singles. Retrieved January 12, 2012.
"ARIA Charts – Accreditations – 2024 Singles" (PDF). Australian Recording Industry Association. Retrieved June 14, 2024.
Japan Hot 100: 2009/07/06付け. Billboard Japan (in Japanese). Archived from the original on December 22, 2015. Retrieved December 22, 2015.
"Taylor Swift: Artist Chart History". Official Charts Company. Retrieved January 12, 2012.
"The Irish Charts – Search Results – Love Story". Irish Singles Chart. Retrieved January 23, 2020.
"Taylor Swift – Love Story". Top 40 Singles. Retrieved January 12, 2012.
"Taylor Swift Chart History (Canadian Hot 100)". Billboard. Retrieved October 21, 2022.
"Official Scottish Singles Sales Chart Top 100". Official Charts Company. Retrieved August 9, 2020.
"Hits of the World". Billboard. Vol. 121, no. 10. March 14, 2009. p. 54. Archived from the original on February 7, 2023. Retrieved November 28, 2021.
"Love Story – Radio Top100 Oficiální" (in Czech). International Federation of the Phonographic Industry. Archived from the original on July 18, 2011. Retrieved March 6, 2011.
"Archívum – Slágerlisták – MAHASZ" (in Hungarian). Single (track) Top 40 lista. Magyar Hanglemezkiadók Szövetsége. Retrieved December 29, 2021.
"Taylor Swift – Love Story". VG-lista. Retrieved January 12, 2012.
"Taylor Swift – Love Story". Singles Top 100. Retrieved January 12, 2012.
"British single certifications – Taylor Swift – Love Story". British Phonographic Industry. Retrieved April 21, 2023.
"New Zealand single certifications – Taylor Swift – Love Story". Radioscope. Retrieved December 19, 2024. Type Love Story in the "Search:" field.
"Canadian single certifications – Taylor Swift – Love Story". Music Canada.
"Danish single certifications – Taylor Swift – Love Story". IFPI Danmark. Retrieved May 23, 2022.
"Gold-/Platin-Datenbank (Taylor Swift; 'Love Story')" (in German). Bundesverband Musikindustrie. Retrieved June 4, 2023.
"Japanese digital single certifications – Taylor Swift – Love Story" (in Japanese). Recording Industry Association of Japan. Retrieved September 24, 2019. Select 2019年8月 on the drop-down menu
"Digital Music Sales Around the World" (PDF). International Federation of the Phonographic Industry. January 21, 2010. Archived from the original (PDF) on January 22, 2012. Retrieved March 6, 2011.
Stefano, Angela (February 12, 2021). "Taylor Swift's New 'Love Story' Leads Re-Recorded Fearless". Taste of Country. Archived from the original on November 18, 2022. Retrieved November 13, 2021.
"On the Set Behind the Scenes at 'Love Story'". Taylor Swift: On the Set. 22:00 minutes in. Great American Country.
Lewis, Randy (October 16, 2008). "Who's That Romeo in Taylor Swift's 'Love Story' Video?". Los Angeles Times. Archived from the original on February 6, 2011. Retrieved February 17, 2011.
Bierly, Mandi (September 17, 2008). "How Much Do You Love Taylor Swift's 'Love Story' Video?". Entertainment Weekly. Archived from the original on December 5, 2011. Retrieved February 17, 2011.
Ross, Rebecca (September 16, 2008). "Super (Lucky) Fan". Pensacola News Journal. p. B1. ProQuest 436152130.
"Taylor Swift Premiering New Video on CMT". CMT. September 12, 2008. Archived from the original on June 6, 2011. Retrieved February 16, 2011.
"On Tap: Cable's New Shows". Multichannel News. Vol. 29, no. 42. October 27, 2008. ProQuest 219862324.
"30 Taylor Swift Music Videos, Ranked". Spin. November 12, 2017. Archived from the original on November 13, 2017. Retrieved August 18, 2022.
LeSavage, Halie (October 27, 2017). "How Taylor Swift Is Counting on Fashion to Change Her Reputation". Glamour. Archived from the original on July 23, 2021. Retrieved August 19, 2022.
Roland, Tom (October 23, 2010). "Princess Superstar". Billboard. Vol. 122, no. 39. p. 21. Archived from the original on February 7, 2023. Retrieved September 18, 2022 – via Google Books.
"BMI Country Awards 2009 Big Winners". Broadcast Music Incorporated. Archived from the original on May 28, 2011. Retrieved March 12, 2011.
Richards, Kevin (November 11, 2009). "Taylor Swift's 'Love Story' Named Song Of The Year At BMI Country Awards". American Songwriter. Archived from the original on November 1, 2022. Retrieved September 20, 2022.
Titus, Christa (April 3, 2010). "Backbeat". Billboard. Vol. 122, no. 13. p. 58. Archived from the original on February 7, 2023. Retrieved August 27, 2022 – via Google Books.
"APRA 2010 nominations list". The Sydney Morning Herald. May 25, 2010. Archived from the original on February 8, 2016. Retrieved January 25, 2015.
"People's Choice Awards 2009 Nominees". People's Choice Awards. Archived from the original on October 27, 2009. Retrieved March 11, 2011.
Knox, David (September 20, 2009). "2009 Kid's Choice Awards: Nominees". TV Tonight. Archived from the original on April 22, 2016. Retrieved March 11, 2011.
"Teen Choice Awards 2009 nominees". The Los Angeles Times. June 15, 2009. Archived from the original on January 11, 2016. Retrieved March 12, 2011.
"Teen Choice Awards 2009 Music". Teen Choice Awards. Archived from the original on October 19, 2009. Retrieved October 19, 2009.
"Nominations Announced for the 44th Annual Academy of Country Music Awards" (Press release). Academy of Country Music. Archived from the original on March 11, 2011. Retrieved March 11, 2011.
"Winners Announced for the 44th Annual Academy of Country Music Awards" (Press release). Academy of Country Music. Archived from the original on July 7, 2011. Retrieved March 11, 2011.
"2009 CMT Music Awards: Winners". CMT. Archived from the original on December 7, 2012. Retrieved March 11, 2011.
"2009 CMA Awards: Winners". CMT. Archived from the original on July 22, 2012. Retrieved March 11, 2011.
"The MYX Music Awards 2010 Winners". MYX. Archived from the original on November 14, 2011. Retrieved March 11, 2011.
Vaziri, Aidin; Irshad, Zara; Williams, Andrew (July 26, 2023). "What Songs Will Taylor Swift Play During the Eras Tour? Here's What Set List Data Reveals". San Francisco Chronicle. Archived from the original on July 31, 2023. Retrieved July 31, 2023.
Zaleski 2024, p. 32.
Vena, Jocelyn (October 26, 2010). "Taylor Swift Shines During Today Show Set". MTV News. Archived from the original on June 29, 2011. Retrieved March 13, 2011.
Spencer 2010, p. 65.
Keel, Beverly (November 7, 2008). "Ellen Throws On-Air Album Party for Taylor". The Tennessean. ProQuest 239917843.
Bonaguro, Alison (January 12, 2009). "Taylor Swift Should've Had More Banjo on Saturday Night Live". CMT. Archived from the original on January 18, 2009. Retrieved March 11, 2011.
Spencer 2010, p. 136.
Chancellor, Jennifer (September 14, 2009). "Family Fun: Music". Tulsa World. p. D3. ProQuest 395461704.
Goodspeed, John (February 5, 2009). "Hats Off to the Entertainers". San Antonio Express-News. p. G.6.
Ross, Curtis (March 2, 2009). "Swift, Fans Bond Over Boys, Heartache". Tampa Tribune. p. 2.
McCabe, Kathy (March 14, 2009). "Souvenir Concert Guide". The Daily Telegraph. Australia. p. 2. ProQuest 359873426.
"Sunday Night's LP Field Show: Taylor Puts Forth Her Truth, Chesney Puts On a High-Energy Closer and John Rich Puts On a Fur Coat". The Tennessean. June 14, 2009. ProQuest 239936915.
DeDekker, Jeff (July 11, 2009). "Swept Off Their Feet; Taylor Swift Simply Scintillating". Regina Leader-Post. p. A1. ProQuest 350032490.
Fischer, Reed (March 8, 2010). "Concert Review: Oscar-less Taylor Swift Still Wins Over BankAtlantic Center on March 7". Miami New Times. Archived from the original on March 4, 2011. Retrieved March 13, 2011.
McDonnel, Brandy (April 1, 2010). "Concert Review: Taylor Swift Brings Fearless Show to Ford Center". The Oklahoman. Archived from the original on July 7, 2012. Retrieved May 21, 2010.
Pareles, Jon (August 28, 2009). "She's a Little Bit Country, a Little Bit Angry". The New York Times. Archived from the original on March 22, 2012. Retrieved March 13, 2011.
Frehsee, Nicole (August 28, 2009). "Taylor Swift Performs a Fearless Set at Madison Square Garden". Rolling Stone. Archived from the original on November 6, 2022. Retrieved September 1, 2022.
Murray, Nick (October 13, 2011). "The Top 5 Moments From Taylor Swift's New Journey To Fearless DVD". The Village Voice. Archived from the original on November 5, 2022. Retrieved October 13, 2011.
Zachariah, Natasha Ann (February 11, 2011). "Scream for Swift". The Straits Times. Singapore. ProQuest 852596571.
Jenkin, Lydia (March 17, 2012). "Concert Review: Taylor Swift at Vector Arena". The New Zealand Herald. Archived from the original on November 30, 2021. Retrieved August 19, 2022.
Griffith, Carson (October 8, 2012). "Taylor-Made for Attention". New York Daily News. p. 17. ProQuest 1095141571.
Willman, Chris (October 17, 2012). "Taylor Swift Tapes VH1 Storytellers, Lifts Curtain on New Songs from Red". The Hollywood Reporter. Archived from the original on July 27, 2021. Retrieved August 19, 2022.
Lansky, Sam (January 25, 2013). "Taylor Swift Performs "Love Story" & "We Are Never Ever Getting Back Together" At 40 Principales: Watch". Idolator. Archived from the original on January 13, 2016. Retrieved January 26, 2013.
Semon, Craig S. (July 27, 2013). "Taylor Swift Red Hot in Foxboro". Telegram & Gazette. Archived from the original on September 22, 2022. Retrieved September 20, 2022.
Lipshutz, Jason (September 20, 2014). "Taylor Swift Shakes Off the 'Frenemies' During iHeartRadio Fest Performance: Watch". Billboard. Archived from the original on September 20, 2014. Retrieved September 20, 2014.
Masley, Ed (August 18, 2015). "Taylor Swift Shakes Off Her Country Roots on '1989' Tour". The Arizona Republic. Archived from the original on January 6, 2021. Retrieved August 18, 2015.
Sheffield, Rob (July 11, 2015). "Taylor Swift's Epic '1989' Tour". Rolling Stone. Archived from the original on April 13, 2020. Retrieved September 20, 2022.
Sheffield, Rob (May 9, 2018). "Rob Sheffield Reviews Taylor Swift's 'Reputation' Tour Kickoff". Rolling Stone. Archived from the original on September 12, 2018. Retrieved September 12, 2018.
Brandle, Lars (April 24, 2019). "Taylor Swift Took Some of the World's Biggest Stars Down Memory Lane With This Performance". Billboard. Archived from the original on April 24, 2019. Retrieved April 24, 2019.
Mylrea, Hannah (September 10, 2019). "Taylor Swift's The City of Lover concert: a triumphant yet intimate celebration of her fans and career". NME. Archived from the original on September 16, 2019. Retrieved September 12, 2019.
Gracie, Bianca (November 24, 2019). "Taylor Swift Performs Major Medley Of Hits, Brings Out Surprise Guests For 'Shake It Off' at 2019 AMAs". Billboard. Archived from the original on November 26, 2019. Retrieved November 28, 2019.
Willman, Chris (July 21, 2022). "Taylor Swift and Haim Join Forces to Pour 'Gasoline' on 'Love Story' at London Concert". Variety. Archived from the original on November 22, 2022. Retrieved November 22, 2022.
Willman, Chris (March 18, 2023). "Taylor Swift's 'Eras' Show Is a Three-Hour, 44-Song Epic That Leaves 'Em Wanting More: Concert Review". Variety. Archived from the original on March 18, 2023. Retrieved March 26, 2023.
Anderson, Kyle (June 17, 2009). "Taylor Swift Raps 'Thug Story' With T-Pain On CMT Awards". MTV News. Archived from the original on December 13, 2022. Retrieved September 21, 2022.
Leight, Elias (July 30, 2020). "Taylor Swift Has 16 New Songs, But an Old One Is Her TikTok Hit". Rolling Stone. Archived from the original on August 1, 2020. Retrieved August 18, 2020.
"Listy bestsellerów, wyróżnienia :: Związek Producentów Audio-Video Archived October 29, 2020, at the Wayback Machine". Polish Airplay Top 100. Retrieved October 26, 2020.
Chelosky, Danielle (August 21, 2024). "Coldplay's Chris Martin & Maggie Rogers Cover Taylor Swift In Vienna After Her Canceled Tour Dates". Stereogum. Retrieved August 22, 2024.
"Taylor Swift – Love Story" (in German). Ö3 Austria Top 40. Retrieved January 12, 2012.
"Taylor Swift – Love Story" (in Dutch). Ultratip. Retrieved January 12, 2012.
"Taylor Swift – Love Story" (in French). Ultratop 50. Retrieved January 12, 2012.
"Taylor Swift Chart History (Canada AC)". Billboard. Retrieved October 21, 2022.
"Taylor Swift Chart History (Canada CHR/Top 40)". Billboard. Retrieved January 12, 2012.
"Taylor Swift Chart History (Canada Country)". Billboard. Retrieved January 12, 2012.
"Taylor Swift Chart History (Canada Hot AC)". Billboard. Retrieved January 12, 2012.
Taylor Swift — Love Story. TopHit. Retrieved April 19, 2021.
"Taylor Swift – Love Story". Tracklisten. Retrieved January 12, 2012.
"Taylor Swift Chart History (Euro Digital Song Sales)". Billboard. Retrieved March 12, 2018.
"Taylor Swift: Love Story" (in Finnish). Musiikkituottajat. Retrieved December 3, 2014.
"Taylor Swift – Love Story" (in French). Les classement single. Retrieved January 12, 2012.
"Taylor Swift – Love Story" (in German). GfK Entertainment charts. Retrieved January 12, 2012.
"Japan Adult Contemporary Airplay Chart". Billboard Japan (in Japanese). Retrieved October 31, 2023.
"Taylor Swift Chart History (Mexico Ingles Airplay)". Billboard. Archived from the original on November 1, 2022. Retrieved January 6, 2022.
"Taylor Swift – Love Story" (in Dutch). Single Top 100. Retrieved January 12, 2012.
"ČNS IFPI" (in Slovak). Hitparáda – Radio Top 100 Oficiálna. IFPI Czech Republic. Note: insert 200915 into search.
"Taylor Swift – Love Story" Canciones Top 50. Retrieved January 12, 2012.
"Taylor Swift – Love Story". Swiss Singles Chart. Retrieved January 12, 2012.
"Taylor Swift Chart History (Latin Pop Songs)". Billboard. Retrieved January 12, 2012.
"Taylor Swift – Billboard Singles". AllMusic. Archived from the original on December 28, 2010. Retrieved December 28, 2010.
"TOP 20 Most Streamed International Singles In Malaysia Week 10 (01/03/2024-07/03/2024)". Recording Industry Association of Malaysia. March 16, 2024. Retrieved March 16, 2024 – via Facebook.
"Taylor Swift – Love Story". AFP Top 100 Singles. Retrieved June 12, 2024.
"RIAS Top Charts Week 10 (1 – 7 Mar 2024)". RIAS. Archived from the original on March 12, 2024. Retrieved March 12, 2024.
"Best of 2008 – Hot 100 Songs". Billboard. Archived from the original on April 1, 2019. Retrieved March 6, 2011.
"Year End Charts – Hot Country Songs – Issue Date: 2008". Billboard. Archived from the original on July 5, 2011. Retrieved August 15, 2011.
"ARIA Charts – End Of Year Charts – Top 100 Singles 2009". Australian Recording Industry Association. Archived from the original on March 5, 2016. Retrieved March 13, 2010.
"Best of 2009 – Canadian Hot 100 Songs". Billboard. Archived from the original on May 25, 2013. Retrieved March 6, 2011.
"2009 Year End Charts – European Hot 100 Singles". Billboard. Archived from the original on October 4, 2012. Retrieved September 30, 2013.
"Top de l'année Top Singles 2009" (in French). Syndicat National de l'Édition Phonographique. Archived from the original on February 13, 2021. Retrieved August 27, 2020.
"Year End Charts – Japan Hot 100". Billboard. Archived from the original on October 7, 2012. Retrieved June 15, 2011.
"RIANZ Annual Top 50 Singles Chart 2008 (see '2009 – Singles')". Recording Industry Association of New Zealand. Archived from the original on September 15, 2008. Retrieved March 13, 2010.
"Årslista Singlar – År 2009". Swedish Recording Industry Association. Archived from the original on July 19, 2011. Retrieved August 8, 2018.
"UK Year-end Songs 2009" (PDF). ChartsPlus. The Official Charts Company. p. 5. Archived (PDF) from the original on August 21, 2010. Retrieved June 14, 2010.
"Best of 2009 – Hot 100 Songs". Billboard. Archived from the original on April 14, 2013. Retrieved March 6, 2011.
"Adult Contemporary Songs: Page 1". Billboard. Archived from the original on July 10, 2015. Retrieved July 28, 2015.
"Adult Pop Songs: Page 1". Billboard. Archived from the original on July 5, 2015. Retrieved July 28, 2015.
"Pop Songs: 2009 Year-End Charts". Billboard. Archived from the original on October 14, 2015. Retrieved July 28, 2015.
McCabe, Kathy (January 7, 2010). "Delta Goodrem's talents top the charts". The Daily Telegraph. Archived from the original on January 9, 2010. Retrieved July 29, 2010.
"Best of 2000s – Hot 100 Songs". Billboard. Archived from the original on March 29, 2011. Retrieved March 6, 2011.
"Austrian single certifications – Taylor Swift – Love Story" (in German). IFPI Austria. Retrieved May 29, 2024.
"Brazilian single certifications – Taylor Swift – Love Story" (in Portuguese). Pro-Música Brasil. Retrieved May 1, 2024.
"Italian single certifications" (in Italian). Federazione Industria Musicale Italiana. Retrieved June 26, 2023. Select "2023" in the "Anno" drop-down menu. Select "Singoli" under "Sezione".
"Spanish single certifications – Taylor Swift – Love Story". El portal de Música. Productores de Música de España. Retrieved April 1, 2024.
"American ringtone certifications – Taylor Swift – Love Story". Recording Industry Association of America. Retrieved July 7, 2022.
"Love Story (Pop Mix) – Single by Taylor Swift". iTunes Store. Archived from the original on February 19, 2011. Retrieved February 19, 2011.
"Love Story (Stripped) – Single by Taylor Swift". iTunes Store. Archived from the original on June 6, 2015. Retrieved February 19, 2011.
"Love Story – Single by Taylor Swift" (in Japanese). iTunes Store. Archived from the original on January 22, 2015. Retrieved January 20, 2015.
"Love Story – Taylor Swift". AllMusic. Archived from the original on December 7, 2022. Retrieved August 27, 2022.
"Love Story (2009)". 7digital. Archived from the original on April 1, 2013. Retrieved April 1, 2013.
"Love Story" (in German). Universal Music Group. Archived from the original on December 7, 2022. Retrieved August 27, 2022.
Melas, Chloe (November 16, 2020). "Taylor Swift Speaks Out about Sale of Her Masters". CTV News. Archived from the original on November 18, 2020. Retrieved November 19, 2020.
"Taylor Swift Wants to Re-record Her Old Hits". BBC News. August 22, 2019. Archived from the original on August 22, 2019. Retrieved August 22, 2019.
Finnis, Alex (November 17, 2020). "Taylor Swift Masters: The Controversy around Scooter Braun Selling the Rights to Her Old Music Explained". i. Archived from the original on February 12, 2021. Retrieved February 13, 2021.
Shah, Neil (April 9, 2021). "Taylor Swift Releases New Fearless Album, Reclaiming Her Back Catalog". The Wall Street Journal. Archived from the original on October 8, 2021. Retrieved September 25, 2022.
Shaffer, Claire (December 2, 2020). "Taylor Swift Drops Her First Re-Recorded Song – in an Ad for Match". Rolling Stone. Archived from the original on February 13, 2021. Retrieved February 12, 2021.
Lipshutz, Jason (February 11, 2021). "Taylor Swift Announces Re-Recorded Fearless Album: Updated 'Love Story' Out Tonight". Billboard. Archived from the original on February 13, 2021. Retrieved February 11, 2021.
Willman, Chris (February 11, 2021). "Taylor Swift Sets Fearless: Taylor's Version as First in Her Series of Full-Album Do-Overs". Variety. Archived from the original on February 11, 2021. Retrieved February 11, 2021.
Legaspi, Claire Shaffer (February 12, 2021). "Taylor Swift Releases Lyric Video for Re-Recorded 'Love Story'". Rolling Stone. Archived from the original on February 13, 2021. Retrieved February 12, 2021.
Kaufman, Gil (March 26, 2021). "Taylor Swift Surprise Releases Dancefloor 'Elvira Remix' of 'Love Story'". Billboard. Archived from the original on November 12, 2021. Retrieved April 13, 2021.
Willman, Chris (February 12, 2021). "Taylor Swift Brought Back Some Original 'Love Story' Musicians for the Remake: Who Returned and Who Didn't". Variety. Archived from the original on November 30, 2022. Retrieved August 27, 2022.
Wood, Mikael (February 12, 2021). "Taylor Swift's Remade 'Love Story (Taylor's Version)' Is Still a Classic, Just Now All Hers". Los Angeles Times. Archived from the original on February 12, 2021. Retrieved February 12, 2021.
Savage, Mark (February 12, 2021). "Taylor Swift's Two Versions of Love Story Compared". BBC News. Archived from the original on February 13, 2021. Retrieved February 12, 2021.
Li, Shirley (February 13, 2021). "The Old Taylor Swift Never Left". The Atlantic. Archived from the original on June 19, 2021. Retrieved February 13, 2021.
Blackwelder, Carlson; Messer, Lesley (November 25, 2020). "Taylor Swift Talks Re-Recording Old Songs like 'Love Story,' New folklore Concert Film on Disney+". Good Morning America. Archived from the original on December 7, 2022. Retrieved November 4, 2022.
Jagota, Vrinda (February 13, 2021). "Taylor Swift – 'Love Story (Taylor's Version)'". Pitchfork. Archived from the original on May 9, 2021. Retrieved February 13, 2021.
Mylrea, Hannah (February 12, 2021). "Taylor Swift's Re-Recorded 'Love Story (Taylor's Version)' Celebrates Her Fearless Era". NME. Archived from the original on February 13, 2021. Retrieved February 12, 2021.
Vozick-Levinson, Simon (February 12, 2021). "'Love Story (Taylor's Version)' Is a Brilliantly Bittersweet Update on a Classic". Rolling Stone. Archived from the original on February 12, 2021. Retrieved February 13, 2021.
Christgau, Robert (February 17, 2021). "Xgau Sez: February, 2021". And It Don't Stop. Substack. Archived from the original on November 9, 2022. Retrieved April 17, 2021.
Asker, Jim (February 22, 2021). "Taylor Swift's 'Love Story (Taylor's Version)' Debuts at No. 1 on Hot Country Songs Chart: 'I'm So Grateful to the Fans'". Billboard. Archived from the original on December 1, 2022. Retrieved February 22, 2021.
"Top 20 Most Streamed International & Domestic Singles in Malaysia". Archived from the original on July 18, 2021. Retrieved March 6, 2021 – via Facebook.
"Official Irish Singles Chart Top 50". Official Charts Company. Retrieved February 19, 2021.
"RIAS Top Charts". Recording Industry Association Singapore. February 23, 2021. Archived from the original on December 26, 2021. Retrieved March 2, 2021.
"Official Singles Chart Top 100". Official Charts Company. Retrieved February 20, 2021.
"British single certifications – Taylor Swift – Love Story (Taylor's Version)". British Phonographic Industry. Retrieved July 27, 2024.
"Taylor Swift – Love Story (Taylor's Version)". Top 40 Singles. Retrieved February 20, 2021.
Knopper, Steve (October 22, 2021). "Radio Isn't Buying Taylor Swift's Retold 'Love Story'". Billboard. Archived from the original on November 1, 2022. Retrieved September 20, 2022.
Aniftos, Rania (April 12, 2022). "Taylor Swift Earns Her 8th CMT Award With 'Love Story (Taylor's Version)'". Billboard. Archived from the original on November 1, 2022. Retrieved September 25, 2022.
"Taylor Swift – Love Story (Taylor's Version)". ARIA Top 50 Singles. Retrieved February 20, 2021.
"Taylor Swift – Love Story (Taylor's Version)" (in Dutch). Ultratip. Retrieved March 5, 2021.
"Taylor Swift Chart History (Euro Digital Song Sales)". Billboard. Retrieved November 18, 2021.
"Taylor Swift Chart History (Global 200)". Billboard. Retrieved February 23, 2020.
"EHR Top 40 – 2021.03.05". European Hit Radio. Archived from the original on November 13, 2021. Retrieved August 30, 2021.
"Netherlands Single Tip Chart". MegaCharts. February 20, 2021. Archived from the original on April 13, 2021. Retrieved February 20, 2021.
"Taylor Swift – Love Story". AFP Top 100 Singles. Retrieved March 3, 2021.
"Veckolista Singlar, vecka 7" (in Swedish). Sverigetopplistan. Archived from the original on April 13, 2021. Retrieved February 19, 2021.
"Taylor Swift Chart History (Country Airplay)". Billboard. Retrieved April 13, 2021.
"Rolling Stone Top 100, February 12 – February 18, 2021". Rolling Stone. Archived from the original on February 22, 2021. Retrieved February 25, 2021.
"Hot Country Songs – Year-End 2021". Billboard. Archived from the original on December 3, 2021. Retrieved December 2, 2021.
"Brazilian single certifications – Taylor Swift – Love Story (Taylor's Version)" (in Portuguese). Pro-Música Brasil. Retrieved July 23, 2024.
"New Zealand single certifications – Taylor Swift – Love Story (Taylor's Version)". Radioscope. Retrieved December 19, 2024. Type Love Story (Taylor's Version) in the "Search:" field.
"OLiS - oficjalna lista wyróżnień" (in Polish). Polish Society of the Phonographic Industry. Retrieved January 24, 2024. Click "TYTUŁ" and enter Love Story (Taylor's Version) in the search box.
"Spanish single certifications – Taylor Swift – Love Story (Taylor's Version)". El portal de Música. Productores de Música de España. Retrieved April 1, 2024.
"Love Story (Taylor's Version) – Single". Spotify. February 12, 2021. Archived from the original on February 12, 2021. Retrieved February 17, 2021.

    "Love Story (Taylor's Version) [Elvira Remix] – Single". Spotify. March 23, 2021. Archived from the original on March 26, 2021. Retrieved March 26, 2021.

Cited sources

    Barclay, Katie (2018). "Love and Violence in the Music of Late Modernity". Popular Music and Society. 41 (5): 539–555. doi:10.1080/03007766.2017.1378526. hdl:2440/114169. S2CID 148714609.
    Perone, James E. (2017). "Becoming Fearless". The Words and Music of Taylor Swift. The Praeger Singer-Songwriter Collection. ABC-Clio. pp. 5–25. ISBN 978-1-44-085294-7.
    Sloan, Nate; Harding, Charlie; Gottlieb, Iris (2019). "A Star's Melodic Signature: Melody: Taylor Swift—'You Belong with Me'". Switched on Pop: How Popular Music Works, and Why it Matters. Oxford University Press. pp. 21–35. ISBN 978-0-19-005668-1.
    Sloan, Nate (2021). "Taylor Swift and the Work of Songwriting". Contemporary Music Review. 40 (1): 11–26. doi:10.1080/07494467.2021.1945226. S2CID 237695045.
    Spencer, Liv (2010). Taylor Swift: Every Day Is a Fairytale – The Unofficial Story. ECW Press. ISBN 978-1-55022-931-8.
    Tuan, Iris H. (2020). "Shakespeare and Popular Culture: Romeo and Juliet in Film and Pop Music". Pop with Gods, Shakespeare, and AI. Springer Nature. pp. 9–37. ISBN 978-981-15-7297-5.
    Watson, Robert N. (2015). "Lord Capulet's Lost Compromise: A Tragic Emendation and the Binary Dynamics of Romeo and Juliet". Renaissance Drama. 43 (1): 53–84. doi:10.1086/680449. S2CID 194162688.
    Zaleski, Annie (2024). "The Fearless Era". Taylor Swift: The Stories Behind the Songs. Thunder Bay Press. pp. 27–52. ISBN 978-1-6672-0845-9.

External links
Listen to this article (40 minutes)
Duration: 39 minutes and 45 seconds.39:45
Spoken Wikipedia icon
This audio file was created from a revision of this article dated 25 August 2023, and does not reflect subsequent edits.
(Audio help · More spoken articles)

    vte

Taylor Swift songs

    Singles discography

Taylor Swift	

    "Tim McGraw" "Picture to Burn" "Teardrops on My Guitar" "A Place in This World" "Should've Said No" "Our Song"

Fearless	

    "Fearless" "Fifteen" "Love Story" "Hey Stephen" "White Horse" "You Belong with Me" "Breathe" "You're Not Sorry" "The Way I Loved You" "Forever & Always" "The Best Day" "Change"

Taylor's Version	

    "You All Over Me" "Mr. Perfectly Fine" "That's When"

Speak Now	

    "Mine" "Sparks Fly" "Back to December" "Speak Now" "Dear John" "Mean" "The Story of Us" "Never Grow Up" "Enchanted" "Better than Revenge" "Innocent" "Haunted" "Long Live" "Ours"

Taylor's Version	

    "Electric Touch" "When Emma Falls in Love" "I Can See You" "Castles Crumbling"

Red	

    "State of Grace" "Red" "Treacherous" "I Knew You Were Trouble" "All Too Well" "22" "We Are Never Ever Getting Back Together" "The Last Time" "Holy Ground" "Everything Has Changed" "Begin Again"

Taylor's Version	

    "Nothing New" "Message in a Bottle" "I Bet You Think About Me" "Forever Winter"

1989	

    "Welcome to New York" "Blank Space" "Style" "Out of the Woods" "All You Had to Do Was Stay" "Shake It Off" "I Wish You Would" "Bad Blood" "Wildest Dreams" "How You Get the Girl" "This Love" "I Know Places" "Clean" "You Are in Love" "New Romantics"

Taylor's Version	

    "'Slut!'" "Say Don't Go" "Now That We Don't Talk" "Suburban Legends" "Is It Over Now?"

Reputation	

    "...Ready for It?" "End Game" "I Did Something Bad" "Don't Blame Me" "Delicate" "Look What You Made Me Do" "So It Goes..." "Gorgeous" "Getaway Car" "Dress" "Call It What You Want" "New Year's Day"

Lover	

    "I Forgot That You Existed" "Cruel Summer" "Lover" "The Man" "The Archer" "Miss Americana & the Heartbreak Prince" "Paper Rings" "Cornelia Street" "Death by a Thousand Cuts" "London Boy" "Soon You'll Get Better" "False God" "You Need to Calm Down" "Me!"

Folklore	

    "The 1" "Cardigan" "The Last Great American Dynasty" "Exile" "My Tears Ricochet" "Mirrorball" "Seven" "August" "This Is Me Trying" "Illicit Affairs" "Invisible String" "Mad Woman" "Epiphany" "Betty" "Peace" "Hoax" "The Lakes"

Evermore	

    "Willow" "Champagne Problems" "'Tis the Damn Season" "Tolerate It" "No Body, No Crime" "Happiness" "Dorothea" "Coney Island" "Long Story Short" "Marjorie"

Midnights	

    "Lavender Haze" "Maroon" "Anti-Hero" "Snow on the Beach" "You're on Your Own, Kid" "Midnight Rain" "Question...?" "Vigilante Shit" "Bejeweled" "Labyrinth" "Karma" "Sweet Nothing" "Mastermind" "Hits Different" "Bigger Than the Whole Sky" "Would've, Could've, Should've" "You're Losing Me"

The Tortured Poets
Department	

    "Fortnight" "The Tortured Poets Department "My Boy Only Breaks His Favorite Toys" "Down Bad" "So Long, London" "But Daddy I Love Him" "Fresh Out the Slammer" "Florida!!!" "Guilty as Sin?" "Who's Afraid of Little Old Me?" "Loml" "I Can Do It with a Broken Heart" "The Smallest Man Who Ever Lived" "The Alchemy" "Clara Bow" "The Black Dog" "So High School" "Thank You Aimee"

Soundtrack songs	

    "Crazier" "Today Was a Fairytale" "Safe & Sound" "Eyes Open" "Sweeter than Fiction" "I Don't Wanna Live Forever" "Beautiful Ghosts" "Only the Young" "Carolina"

Featured songs	

    "Two Is Better Than One" "Half of My Heart" "Both of Us" "Babe" "Gasoline" "Renegade" "The Joker and the Queen" "The Alcott" "Us"

Other songs	

    "Best Days of Your Life" "Ronan" "Highway Don't Care" "Better Man" "Christmas Tree Farm" "All of the Girls You Loved Before"

    Category

    vte

William Shakespeare's Romeo and Juliet

    vte

CMA Video of the Year
Authority control databases Edit this at Wikidata	

    MusicBrainz work

Categories:

    2008 songs2008 singles2021 singlesTaylor Swift songsMusic videos directed by Trey FanjoyNumber-one singles in AustraliaSongs written by Taylor SwiftSong recordings produced by Taylor SwiftSong recordings produced by Nathan Chapman (record producer)Song recordings produced by Chris RoweBig Machine Records singlesMusic based on Romeo and JulietRepublic Records singlesCountry pop songs

    This page was last edited on 11 January 2025, at 07:57 (UTC).
    Text is available under the Creative Commons Attribution-ShareAlike 4.0 License; additional terms may apply. By using this site, you agree to the Terms of Use and Privacy Policy. Wikipedia® is a registered trademark of the Wikimedia Foundation, Inc., a non-profit organization.

    Privacy policy
    About Wikipedia
    Disclaimers
    Contact Wikipedia
    Code of Conduct
    Developers
    Statistics
    Cookie statement
    Mobile view

    Wikimedia Foundation
    Powered by MediaWiki



Wikipedia The Free Encyclopedia

    Donate
    Create account
    Log in

Contents
(Top)
Background and production

Music and lyrics

Release and commercial performance

Critical reception

Music video

    Development and synopsis
    Reception

Awards and nominations

Live performances and other use

Ryan Adams cover

Credits and personnel

Charts

    Weekly charts
    Year-end charts

Certifications

Release history

"Bad Blood (Taylor's Version)"

    Production and reception
    Personnel
    Charts
        Weekly charts
        Year-end charts
    Certifications

See also

Footnotes

References

        Sources

Bad Blood (Taylor Swift song)

    Article
    Talk

    Read
    Edit
    View history

Tools

Appearance
Text

    Small
    Standard
    Large

Width

    Standard
    Wide

Color (beta)

    Automatic
    Light
    Dark

Featured article
From Wikipedia, the free encyclopedia
"Bad Blood"
Cover artwork of "Bad Blood" by Taylor Swift featuring Kendrick Lamar, showing a black and white photo of Swift
Single by Taylor Swift featuring Kendrick Lamar
from the album 1989
Released	May 17, 2015
Studio	

    MXM (Stockholm)
    Conway Recording (Los Angeles)

Genre	

    Pop

Length	3:31
3:19 (remix)
Label	Big Machine
Songwriter(s)	

    Taylor Swift Max Martin Shellback Kendrick Duckworth[a]

Producer(s)	

    Max Martin Shellback Ilya[a]

Taylor Swift singles chronology
"Style"
(2015) 	"Bad Blood"
(2015) 	"Wildest Dreams"
(2015)
Kendrick Lamar singles chronology
"King Kunta"
(2015) 	"Bad Blood"
(2015) 	"Alright"
(2015)
Music video
"Bad Blood" on YouTube

"Bad Blood" is a song by the American singer-songwriter Taylor Swift from her fifth studio album, 1989 (2014). She wrote the song with the Swedish producers Max Martin and Shellback. It is a pop song using keyboards and hip hop–inspired drum beats, and the lyrics are about betrayal by a close friend. A remix featuring the American rapper Kendrick Lamar, with additional lyrics by Lamar and production by the Swedish musician Ilya, was released to radio as 1989's fourth single on May 17, 2015, by Big Machine and Republic Records.

Music critics gave the album version of "Bad Blood" mixed reviews; some described it as catchy and engaging, but others criticized the production as bland and the lyrics repetitive. The remix version received somewhat more positive comments for Lamar's verses, featured among the best songs of 2015 on lists by NME and PopMatters, and received a Grammy nomination for Best Pop Duo/Group Performance. Critics have retrospectively considered "Bad Blood" one of Swift's worst songs. The single reached number one and received multi-platinum certifications in Australia, Canada, and the United States.

Directed by Joseph Kahn and produced by Swift, the music video for "Bad Blood" features an ensemble cast consisting of female singers, actresses, and models. Having a production that resembles sci-fi and action movies, it won the Grammy Award for Best Music Video and MTV Video Music Awards for the Video of the Year and Best Collaboration. Swift performed "Bad Blood" on the 1989 World Tour (2015), the Reputation Stadium Tour (2018), and the Eras Tour (2023–2024). Following a 2019 dispute regarding the ownership of Swift's back catalog, she re-recorded both the album version and the Lamar remix for her 2023 re-recorded album 1989 (Taylor's Version); both re-recordings are subtitled "Taylor's Version".
Background and production

Swift had identified as a country musician until her fourth studio album, Red,[1] which was released on October 22, 2012.[2] Red's eclectic pop and rock styles beyond the country stylings of Swift's past albums led to critics questioning her country-music identity.[3][4] Swift began writing songs for her fifth studio album in mid-2013 while touring.[5] She was inspired by 1980s synth-pop to create her fifth studio album, 1989, which she described as her first "official pop album" and named after her birth year.[6][7] The album makes extensive use of synthesizers, programmed drum machines, and electronic and dance stylings, a stark contrast to the acoustic arrangements of her country-styled albums.[8][9]

Swift and Max Martin served as executive producers of 1989.[10] On the album's standard edition, Martin and Shellback produced 7 out of 13 songs, including "Bad Blood".[11] Swift wrote "Bad Blood" with Martin and Shellback, who both programmed the track and played electronic keyboards on it. The song was recorded by Sam Holland at Conway Recording Studios in Los Angeles, and by Michael Ilbert at MXM Studios in Stockholm, Sweden. The song was mixed by Serban Ghenea at Mixstar Studios in Virginia Beach, Virginia, and mastered by Tom Coyne at Sterling Sound Studio in New York.[11]
Music and lyrics
"Bad Blood"
Duration: 19 seconds.0:19
The album version of "Bad Blood" features stomping drums and pulsing keyboards.
Problems playing this file? See media help.

"Bad Blood" is a pop song with prominent hip hop stylings.[12][13] It incorporates prominent keyboard tones,[14] hip hop beats, and a pulsing bassline.[15] According to Jon Caramanica of The New York Times, the "booming drums" of the song evoked the "Billy Squier ones often sampled in hip-hop".[16] Jem Aswad of Billboard described the production as "simplistic" and compared it to Gwen Stefani's "Hollaback Girl" (2005),[17] The Observer's Kitty Empire likened the "stark beats" to the music of Charli XCX,[18] and NME's Matthew Horton deemed the song a "bitter stomp" that evokes Beastie Boys.[12] The lyrics portray resentment and anger that result from betrayal, through lyrics such as, "These kinda wounds, they last and they last," and "Band-aids don't fix bullet holes/ You say sorry just for show."[15] The refrain consists of repeated phrases, "Now we got bad blood/ You know it used to be mad love."[14] Jon Pareles described Swift's vocals throughout the refrain as tense,[14] while Consequence of Sound's Sasha Geffen wrote that she sang "through gritted teeth".[15]

Some critics interpreted "Bad Blood" to be about a lost love.[19][20] In an interview for the September 2014 cover issue for Rolling Stone, Swift said that the song was about a fellow female artist whom she had thought of as a close friend; she felt betrayed after this person attempted to "sabotage an entire arena tour" by "[hiring] a bunch of people out from under [her]".[21] She wanted to make it clear that it was about losing a friend and not a lover because she "knew people would immediately be going in one direction", referring to how the audience interpreted her songs in association with her love life.[21] The media speculated the subject to be Katy Perry, who had a publicized fallout with Swift after being friends for several years.[22][23][24] In another interview for GQ in October 2015, Swift reaffirmed the theme of lost friendship and responded to the speculation: "I never said anything that would point a finger in the specific direction of one specific person."[25] According to Chuck Klosterman, by clarifying the inspiration behind "Bad Blood" to divert the media from her love life without disclosing the subject, Swift "propagated the existence of a different rumor that offered the added value of making the song more interesting".[25]
Release and commercial performance
Portrait of Kendrick Lamar
Kendrick Lamar featured and wrote his rap verses on the single release of "Bad Blood", which became his first number-one single in the United States.

After 1989 was released on October 27, 2014, "Bad Blood" first charted on the Billboard Hot 100 for two weeks in November 2014 and January 2015, reaching number 78.[26] In May 2015, a remix version featuring the rapper Kendrick Lamar was released as the fourth single to promote 1989.[27] According to Lamar, Swift reached out to him personally and he agreed because they had been fond of each other's music.[28] On the remix, Lamar raps two verses written by himself, and Ilya contributed additional production.[29] Lamar recalled that the collaboration with Swift went smoothly because "the vibe was right"; he finished his verses in a few takes during a studio session in Los Angeles.[30] When Rolling Stone asked him in 2017 whether he was "taking sides in a pop beef", he responded that he was unaware of it.[31]

Big Machine Records released the remix for digital download on May 17, 2015,[32] the same day that the premiere of its music video took place at the Billboard Music Awards.[33] In the United States, Big Machine and Republic Records sent "Bad Blood" to contemporary hit radio on May 19,[34] and to rhythmic radio on June 9, 2015.[35] Universal Music Group released the song to Italian radio on June 12, 2015.[36] "Bad Blood" re-entered the Hot 100 at number 53 upon its single release[26] and reached number one the following week, on the Billboard Hot 100 chart dated June 6, 2015.[37] "Bad Blood" was the third single from 1989 to reach number one, after "Shake It Off" and "Blank Space"; it was Swift's fourth and Lamar's first career number-one Hot 100 single.[37] After one week at number one, it charted at number two for the next five weeks.[38]

On Billboard's airplay charts, "Bad Blood" reached number one on Pop Songs[39] and Adult Pop Songs.[40] On the Pop Songs chart, after it debuted at number 13 and rose to number 9 the following week, the single tied the record for the quickest timeline to enter the top 10.[41] By reaching number one in five weeks, it registered the shortest duration to top the chart since Nelly's "Over and Over" (2004) featuring Tim McGraw, which spent three weeks before ascending to the top.[39] In the week ending July 12, 2015, the single broke the record for the most single-week plays in the Pop Songs chart's 22-year history, surpassing Wiz Khalifa and Charlie Puth's "See You Again" (2015).[40] According to Nielsen SoundScan, "Bad Blood" was the 10th-best-selling song of 2015 in the United States, selling 2.584 million digital copies.[42] The Recording Industry Association of America certified the single six-times platinum for surpassing six million units based on sales and on-demand streams,[43] and the track had sold 3.2 million digital copies in the United States by July 2019.[44]

"Bad Blood" topped the charts in Australia,[45] Canada,[46] New Zealand,[47] and Scotland.[48] It peaked within the top five in South Africa,[49] Lebanon,[50] and the United Kingdom;[51] and the top ten in Hungary, Finland, and Ireland. The single was certified multi-platinum in Australia (eight-times platinum),[52] Brazil (double diamond),[53] Canada (triple platinum),[54] and the United Kingdom (double platinum).[55] It was certified platinum in Austria,[56] Norway,[57] Portugal,[58] and gold in Denmark, Germany, Italy, and New Zealand.[59] In the United Kingdom, the single had sold 373,000 downloads as of July 2021.[60]
Critical reception

"Bad Blood" received mixed reviews, with many critics deeming it the weakest song on 1989.[27] Mike Diver from Clash described it as "a litany of diary-page break-up clichés set to directionless thumps and fuzzes",[61] while Jay Lustig from The Record criticized Swift's delivery as "merely petulant, howling" and the beats as "repetitive".[62] Several critics found the song neither engaging nor distinctive; the musicologist James E. Perone wrote that listeners would not be able to tell if "Bad Blood" was a "Taylor Swift song" because of its composition and vocals.[63] Andrew Unterberger from Spin said that the lyrics were absent of the specificity that had characterized Swift's previous songs,[64] Mikael Wood from the Los Angeles Times thought that its beat was reminiscent of that on Katy Perry's "Roar" (2013),[65] and Lindsay Zoladz from Vulture considered "Bad Blood" an "ironic" song to be taken as a Perry diss track because other album tracks had "the faceless mall-pop" that was "no better" than the worst songs on Perry's album Prism (2013).[66] Retrospective rankings by Rolling Stone's Rob Sheffield[67] and Paste's Jane Song both ranked it as the worst song Swift had released.[68]

In more positive reviews, several critics considered "Bad Blood" one of the highlights of 1989. The Quietus's Amy Pettifier said that it was one of the album tracks "crammed with merit" and called it "all sass and bile",[8] Entertainment Weekly's Adam Markovitz said that the track was a "potential [hit]" as a "chant-along fight song",[69] and the Toronto Star's Ben Rayner praised it as a "proper keeper on delivery" with its "cheerleader-ish shout-along".[70] Consequence of Sound's Sasha Geffen and Drowned in Sound's Robert Leedham found the song to showcase a defiant attitude; the former attributed this to the production elements of hip hop beats and deep bassline: "they let her slice out her words with real anger, not just passive regret",[15] and the latter wrote that it recalled "iconic hardcore bands you've probably never heard of".[71] Revisiting "Bad Blood" in 2023, Amara Sorosiak of American Songwriter regarded it as a career-defining single for Swift, writing that it exemplified the "shout-able, catchy pop" of her pop albums and solidified her "bold image" as an artist in the 2010s decade.[72] For Vulture's Nate Jones, the song represented the peak of Swift's "Max Martin era", with its melody being expertly crafted but lyrics absent of "humanity".[73]

Reviewing the remix version featuring Lamar, August Brown of the Los Angeles Times expressed confusion towards the rapper's appearance and contended that it was a move to garner a mainstream audience after his "epic" album To Pimp a Butterfly (2015). Brown said that while Lamar's delivery was "not at his most fiery", it proved his artistic versatility "from difficult free jazz [...] to the tightest, glossiest pop out there".[74] Slate's Chris Molanphy praised Ilya's production for highlighting the refrain's musical highlights and lauded Lamar's "tongue-tripping turns of phrase", but he contended that the rapper was in "accessible, maximum-pop mode" while he was supposed to be held "to a higher standard".[29] Alexis Petridis of The Guardian dubbed the single "a masterstroke" with "potent and effective" verses from Lamar and an "even more anthemic" chorus compared to the album version.[75] "Bad Blood" featuring Lamar was listed among the best songs of 2015 by NME (11th)[76] and PopMatters (6th).[77]
Music video
Development and synopsis
A shot of the "Bad Blood" music video showing Swift and her crew walking in front of an explosion
Catastrophe's (played by Swift) team in front of an explosion in the music video for "Bad Blood", which was compared to action movies by media publications.

The music video of "Bad Blood" was directed by Joseph Kahn and produced by Swift. Filmed in Los Angeles on April 12, 2015, the video premiered on May 17, 2015, at the Billboard Music Awards.[78] The video features Swift and Lamar alongside an ensemble cast consisting of female singers, actresses, and fashion models who were dubbed by the media as Swift's "squad".[79][80] Each member of the cast chose their character's name.[81] The cast include, in order of appearance: Catastrophe (Swift), Arsyn (Selena Gomez), Welvin da Great (Lamar), Lucky Fiori (Lena Dunham), the Trinity (Hailee Steinfeld), Dilemma (Serayah), Slay-Z (Gigi Hadid), Destructa X (Ellie Goulding), Homeslice (Martha Hunt), Mother Chucker (Cara Delevingne), Cut Throat (Zendaya), The Crimson Curse (Hayley Williams), Frostbyte (Lily Aldridge), Knockout (Karlie Kloss), Domino (Jessica Alba), Justice (Mariska Hargitay), Luna (Ellen Pompeo), and Headmistress (Cindy Crawford).[82]

Set in a fictional London, the video starts with Catastrophe and her partner, Arsyn, fighting off a group of men in a corporate office for a mysterious briefcase.[83] When all of the men are defeated, Arsyn steals the briefcase from Catastrophe's hand and kicks her out of a window, making her fall onto a car.[81] The song then begins, and Catastrophe and her female squad train to exact their revenge.[84] The video concludes with Catastrophe's and Arsyn's teams two teams facing each other, walking in slow motion as an explosion goes off in the background, blotting out the London skyline.[81][84]
Reception

"Bad Blood" broke Vevo's 24-hour viewing record by accumulating 20.1 million views in its first day of release.[85] Media publications compared the video's production to that of blockbuster movies[b] and opined that it resembled action and sci-fi films and series such as Sin City, RoboCop, Tron, Kill Bill, and Mad Max: Fury Road.[c] Erin Strecker of Billboard commented that there were resemblances to the videos of Britney Spears's "Toxic" and "Womanizer", which were both directed by Kahn.[93] Esquire's Matt Miller said that the video depicted a "sci-fi Taylor",[94] and Rolling Stone described it as "futuristic neo-noir".[81] In Consequence, Mary Siroky deemed it the most memorable music video of the 1989 singles and called it "The Avengers of music videos".[95] Spencer Kornhaber of The Atlantic thought otherwise that it did not succeed on a cinematic level because "the editing becomes so hectic that even the barest bones story here is indiscernible".[92]

Some journalists analyzed the video with regards to Swift's celebrity. According to Time's Daniel D'Addario, with "Bad Blood" and the music videos for other 1989 singles, Swift abandoned the "appropriately lo-fi" videos of her country songs to use videos "as a tool to explore various sides of her personality, and create others", accompanying her artistic reinvention to pop music. D'Addario wrote that Swift followed Madonna by "[paring] visual aesthetics with entirely unrelated songs, giving the viewer a whole new thing to talk about", and thus succeeded in promoting herself as "2015's all-around-perfect pop star".[86] In The Washington Post, Emily Yahr commented that by enlisting high-profile celebrities for the video, Swift proved that she was "the most powerful women in show business" who had "access, status and power" to mobilize a big number of celebrities to go against her adversaries.[96]

Several critics commented on the video in the context of feminism. Websites like The Daily Beast and Deadspin criticized the "supposed hypocrisy", citing the alleged feud with Katy Perry.[80] The "squad" was a point of contention: Kornhaber applauded the video as an imagining of an all-female action movie,[92] but Jennifer Gannon from The Irish Times considered Swift's "squad" as a means to build a cult of personality rather than embody female empowerment,[97] an idea corroborated by Eve Barlow of The Times, who described it as "an exclusive, Mean Girls-style clique of perfect, stalk-limbed and shiny-haired clones".[98] Judy L. Isaksen and Nahed Eltantawy—scholars in popular culture and journalism, and Hannelore Roth—a scholar in literature—argued that Swift's idea of feminism was only applicable to famous and wealthy women. According to Isaksen and Eltantawy, fans of Swift were critical of the supposed "embodiment of privilege" despite her efforts to promote a postfeminist "girlfriend culture".[99] Roth added that by casting Lamar as the ringleader behind the female squad, the video was "just a violent, pre-modern copy of the patriarchal structures at the office".[100]
Awards and nominations

At the 2015 MTV Video Music Awards, "Bad Blood" was nominated in eight categories and won in two: Video of the Year and Best Collaboration;[101] it was Swift's first Video of the Year win.[102] The song was nominated for Best Pop Duo/Group Performance and won Best Music Video at the 58th Annual Grammy Awards in 2016.[103] It was recognized as one of the biggest songs of the year at both the ASCAP Pop Music Awards by the American Society of Composers, Authors and Publishers (ASCAP)[104] and the 64th Annual BMI Pop Awards by Broadcast Music, Inc.[105]

"Bad Blood" won fan-voted categories at the Teen Choice Awards (Choice Music – Collaboration),[106] the MTV Europe Music Awards (Best Song),[107] and the Radio Disney Music Awards (Song of the Year, Best Breakup Song).[108] It received nominations at the American Music Awards,[109] the People's Choice Awards,[110] the Nickelodeon Kids' Choice Awards,[111] and the iHeartRadio Music Awards.[112] The music video additionally won accolades at Mexico's Telehit Awards (Video of the Year),[113] the Philippines' Myx Music Award (Favorite International Video),[114] and France's NRJ Music Award (Video of the Year).[115]
Live performances and other use
Swift performing, dressed in a black leather outfit
Swift performing "Bad Blood" on the 1989 World Tour in 2015

At the 2015 MTV Video Music Awards on August 30, Swift and Nicki Minaj jointly performed "Bad Blood" and "The Night Is Still Young".[116] Swift also sang the song during her concerts at the United States Grand Prix on October 22, 2016,[117] and the pre-Super Bowl event Super Saturday Night on February 4, 2017.[118]

On the 1989 World Tour (2015), Swift performed "Bad Blood" wearing a black leather suit as dancers performed acrobatics behind her.[119] She included the song in the set list of the Reputation Stadium Tour (2018), where she performed it in a mash-up with "Should've Said No" (2008), which incorporated a country-influenced guitar riff.[120][121] According to The Ringer's Nora Princiotti, the mash-up improved one of Swift's weakest songs ("Bad Blood") by tweaking its arrangement and using the melody of an "early classic" ("Should've Said No").[122] On the Eras Tour (2023–2024), Swift performed "Bad Blood" as the screen showed a house on fire and the venue lit up in red.[123][124]

"Bad Blood" was parodied in various other mediums. The comedians Cariad Lloyd and Jenny Bede's parody of "Bad Blood" called for withdrawal of taxation of women's sanitary products in the United Kingdom.[125] The animated web series How It Should Have Ended in September 2015 created a parody called "Bat Blood", which satirizes the marketing of the 2016 film Batman v Superman: Dawn of Justice.[126] Kevin McDevitt, a writer and filmmaker, made a parody titled "Good Blood" to encourage viewers become donors for the bone marrow transplant via the non-profit National Marrow Donor Program.[127] The music video was parodied in the sitcom Great News, featuring a "squad" consisting of Tina Fey and Nicole Richie, which aired in October 2017.[128]

The rock band Drenge and the singer-songwriter Alessia Cara covered the song for BBC Radio 1's live sessions in June[129] and July 2015,[130] and the rapper-singer Drake used a snippet of it in an advertisement for Apple Music in November 2016.[131] Anthony Vincent, a YouTuber and musician, covered "Bad Blood" to make it sound like it had been sung by 19 diverse acts, including the Rolling Stones, TLC, Cyndi Lauper, Barney & Friends, and Sepultura.[132]
Ryan Adams cover

Ryan Adams, an American singer-songwriter, covered "Bad Blood" as part of his track-by-track interpretation of Swift's 1989.[133] Adams said that Swift's 1989 helped him cope with emotional hardships and that he wanted to interpret the songs from his perspective "like it was Bruce Springsteen's Nebraska".[134] His version of "Bad Blood" is an alt-country and folk-pop song that uses acoustic guitar strums and live drums.[135][136][137] Prior to his cover album's release, Adams previewed "Bad Blood" on Apple Music's Beats 1 radio and then released it as a single, on September 17, 2015.[138]

Andrew Unterberger from Spin preferred Adams's version to Swift's, writing that it "[strips the] overbearing hyperactivity ... [and removes the] sneering obnoxiousness".[139] Annie Zaleski of The A.V. Club complimented the "[watercolor]-hued strings and well-placed percussion thumps".[140] In less enthusiastic reviews, Billboard's Chris Payne deemed it the worst cover on Adams's 1989 because he thought it failed to highlight Swift's songwriting strengths,[141] and Vulture's Jillian Mapes thought that by switching the "sinister beats" with "coffeehouse-singer [...] strumming and a jangly counter-melody in the chorus", Adams turned "Bad Blood" from a sonically distinctive track into an unoriginal song.[142] His cover peaked at number 25 on the Ultratop chart of Belgian Wallonia[143] and number 36 on Billboard's Rock Airplay chart.[144]
Credits and personnel

Adapted from the liner notes of 1989[11] and Tidal[145]

    Taylor Swift – vocals, backing vocals, songwriter
    Kendrick Lamar[a] – featured vocals, backing vocals, songwriter
    Max Martin – producer, songwriter, programmer, keyboards, piano
    Shellback – backing vocals, producer, songwriter, programmer, acoustic guitar, bass guitar, keyboards, drums, percussion, sounds (stomps and knees)
    Ilya[a] – backing vocals, producer, programmer, recording engineer
    Michael Ilbert – recording engineer
    Sam Holland – recording engineer
    Ben Sedano – assistant recording engineer
    Cory Bice – assistant recording engineer
    Peter Carlsson – Pro Tools engineer
    Serban Ghenea – mixing engineer
    John Hanes – mixer
    Tom Coyne – mastering engineer

Charts
Weekly charts
2015–2016 weekly chart performance for "Bad Blood" Chart (2015–2016) 	Peak
position
Australia (ARIA)[45] 	1
Austria (Ö3 Austria Top 40)[146] 	22
Belgium (Ultratop 50 Flanders)[147] 	26
Belgium (Ultratop 50 Wallonia)[148] 	26
Brazil (Billboard Brasil Hot 100)[149] 	81
Canada (Canadian Hot 100)[46] 	1
Canada AC (Billboard)[150] 	1
Canada CHR/Top 40 (Billboard)[151] 	1
Canada Hot AC (Billboard)[152] 	1
Czech Republic (Rádio – Top 100)[153] 	29
Euro Digital Song Sales (Billboard)[154] 	2
Finland Download (Latauslista)[155] 	5
France (SNEP)[156] 	14
Germany (GfK)[157] 	29
Greece Digital Song Sales (Billboard)[158] 	2
Hungary (Single Top 40)[159] 	10
Ireland (IRMA)[160] 	8
Italy (FIMI)[161] 	93
Japan (Japan Hot 100)[162] 	20
Japan Adult Contemporary (Billboard)[163] 	69
Lebanon (Lebanese Top 20)[50] 	4
Mexico Airplay (Billboard)[164] 	16
Netherlands (Dutch Top 40)[165] 	33
New Zealand (Recorded Music NZ)[47] 	1
New Zealand (Recorded Music NZ)[166]
Solo version 	40
Scotland (OCC)[48] 	1
Slovakia (Rádio Top 100)[167] 	65
South Africa (EMA)[49] 	2
Switzerland (Schweizer Hitparade)[168] 	28
UK Singles (OCC)[51] 	4
US Billboard Hot 100[169] 	1
US Adult Contemporary (Billboard)[170]
Solo version 	9
US Adult Pop Airplay (Billboard)[171]
Solo version 	1
US Dance/Mix Show Airplay (Billboard)[172] 	5
US Dance Club Songs (Billboard)[173] 	37
US Pop Airplay (Billboard)[174] 	1
US Rhythmic (Billboard)[175] 	6
Venezuela (Record Report)[176] 	4
2023 weekly chart performance for "Bad Blood" Chart (2023) 	Peak
position
Global 200 (Billboard)[177] 	198
Portugal (AFP)[178] 	28
Singapore (RIAS)[179] 	13
	
Year-end charts
2015 year-end charts for "Bad Blood" Chart (2015) 	Position
Australia (ARIA)[180] 	18
Canada (Canadian Hot 100)[181] 	11
Hungary (Single Top 40)[182] 	91
UK Singles (Official Charts Company)[183] 	88
US Billboard Hot 100[184] 	15
US Adult Contemporary (Billboard)[185] 	20
US Adult Pop Songs (Billboard)[186] 	11
US Dance/Mix Show Songs (Billboard)[187] 	28
US Pop Songs (Billboard)[188] 	4
US Rhythmic (Billboard)[189] 	33
2016 year-end chart for "Bad Blood" Chart (2016) 	Position
Brazil (Brasil Hot 100)[190] 	60

Certifications
Certifications for "Bad Blood" Region 	Certification 	Certified units/sales
Australia (ARIA)[52] 	8× Platinum 	560,000‡
Austria (IFPI Austria)[56] 	Platinum 	30,000‡
Brazil (Pro-Música Brasil)[53] 	2× Diamond 	500,000‡
Canada (Music Canada)[54] 	3× Platinum 	240,000*
Denmark (IFPI Danmark)[191] 	Gold 	45,000‡
Germany (BVMI)[192] 	Gold 	200,000‡
Italy (FIMI)[193] 	Gold 	50,000‡
New Zealand (RMNZ)[194] 	2× Platinum 	60,000‡
Norway (IFPI Norway)[57] 	Platinum 	60,000‡
Portugal (AFP)[58] 	Platinum 	10,000‡
Spain (PROMUSICAE)[195] 	Gold 	30,000‡
United Kingdom (BPI)[55] 	2× Platinum 	1,200,000‡
United States (RIAA)[43] 	6× Platinum 	6,000,000‡

* Sales figures based on certification alone.
‡ Sales+streaming figures based on certification alone.
Release history
Release dates and formats for "Bad Blood" Region 	Date 	Format(s) 	Version 	Label(s) 	Ref.
Various 	May 17, 2015 	

    Digital downloadstreaming

	Remix featuring Kendrick Lamar 	Big Machine 	[32]
United States 	May 19, 2015 	Contemporary hit radio 	

    Big MachineRepublic

	[34]
June 9, 2015 	Rhythmic radio 	Republic 	[35]
Italy 	June 12, 2015 	Radio airplay 	Universal 	[196]
Original 	[36]
"Bad Blood (Taylor's Version)"
"Bad Blood (Taylor's Version)"
Song by Taylor Swift featuring Kendrick Lamar
from the album 1989 (Taylor's Version)
Released	October 27, 2023
Studio	Prime Recording (Nashville)
Length	3:31
3:20 (remix)
Label	Republic
Songwriter(s)	

    Taylor Swift Kendrick Duckworth[a] Max Martin Shellback

Producer(s)	

    Taylor Swift Christopher Rowe

Lyric video
"Bad Blood (Taylor's Version)" on YouTube

Swift departed from Big Machine and signed a new contract with Republic Records in 2018. She began re-recording her first six studio albums in November 2020.[197] The decision followed a 2019 dispute between Swift and the talent manager Scooter Braun, who acquired Big Machine Records, over the masters of Swift's albums that the label had released.[198][199] By re-recording the albums, Swift had full ownership of the new masters, which enabled her to encourage licensing of her re-recorded songs for commercial use in hopes of substituting the Big Machine-owned masters.[200] She denoted the re-recordings with a "Taylor's Version" subtitle.[201]

The re-recording of "Bad Blood" is titled "Bad Blood (Taylor's Version)". A snippet of it featured in the 2022 animated film DC League of Super-Pets.[202] Although Swift had not re-recorded 1989, she agreed to re-record "Bad Blood" for the film upon request from Season Kent, its music supervisor.[203] The full re-recorded song is included as part of 1989 (Taylor's Version), which was released on October 27, 2023.[204] The remix featuring Lamar was also re-recorded as the bonus track of the deluxe edition of 1989 (Taylor's Version). Swift expressed her gratitude towards Lamar on social media, saying that his participation in the re-recording was "surreal and bewildering".[205]
Production and reception

Swift produced "Bad Blood (Taylor's Version)" with Christopher Rowe, who had produced her previous re-recordings.[206][207] The track was programmed and edited by Derek Garten at Prime Recording in Nashville, and Swift's vocals were recorded by Rowe at Kitty Committee and Electric Lady Studios in New York. Musicians who contributed to the track included Mike Meadows (synth, acoustic guitar), Dan Burns (drums, synth bass, synth), Amos Heller (bass guitar), and Matt Billingslea (drums). Serban Ghenea mixed the song at MixStar Studios in Virginia Beach.[208]

The arrangement of "Bad Blood (Taylor's Version)" remains identical to that of the original version.[209] Some critics commented that there were subtle changes; Notion's Rachel Martin wrote that Swift made "some dialect tweaks" and sang "with more depth and emotion" in the bridge, which resulted in a more powerful conclusion,[210] while The Music's Tione Zylstra said that her vocals were "angrier and bitter".[211] Ed Power of the i described it as a "timeless diss track",[212] and Mark Sutherland of Rolling Stone UK commented the track "remains astounding".[213] Commenting on the re-recorded remix, Elizabeth Braaten of Paste praised Swift and Lamar as "a match made in radio heaven".[209] Giving the track a negative review, Pitchfork's Shaad D'Souza said that it "sounds more basic, bratty, and boring than ever".[207]

On the Billboard Hot 100, "Bad Blood (Taylor's Version)" debuted at number seven on the chart dated November 11, 2023,[214] extending Swift's record for the most top-10 singles (49) among women.[215] On the Billboard Global 200, it debuted at number six; with other 1989 (Taylor's Version) tracks, it helped Swift become the first artist to occupy the entire top six of the Global 200 chart simultaneously.[216] The track also peaked in the top 10 on charts of Canada (7),[217] New Zealand (10),[218] and the Philippines.[219] It was certified gold in Australia[220] and Brazil.[53]
Personnel

Adapted from the liner notes of 1989 (Taylor's Version)[208]

    Taylor Swift – lead vocals, background vocals, songwriting, production
    Christopher Rowe – production, background vocals, vocal engineering
    Mike Meadows – synthesizer, acoustic guitar
    Dan Burns – drum programming, synth bass, synthesizer, additional engineering
    Matt Billingslea – drum programming, drums
    Amos Heller – bass guitar
    Derek Garten – programming, engineering, editing
    Ryan Smith – mastering
    Serban Ghenea – mixing
    Bryce Bordone – mix engineering
    Max Martin – songwriting
    Shellback – songwriting
    Kendrick Lamar – rap vocals, songwriting[a]
    Ilya Salmanzadeh – background vocals[a]

Charts
Weekly charts
Weekly chart performance for "Bad Blood (Taylor's Version)" Chart (2023–2024) 	Peak
position
Australia (ARIA)[221] 	52
Brazil (Brasil Hot 100)[222] 	72
Canada (Canadian Hot 100)[217] 	7
Global 200 (Billboard)[223] 	6
Greece International (IFPI)[224] 	46
Ireland (Billboard)[225] 	13
Malaysia International (RIM)[226] 	19
New Zealand (Recorded Music NZ)[218] 	10
Philippines (Billboard)[219] 	10
Sweden (Sverigetopplistan)[227] 	69
UAE (IFPI)[228] 	20
UK (Billboard)[229] 	13
UK Singles Downloads (OCC)[230] 	10
UK Singles Sales (OCC)[231] 	12
UK Streaming (OCC)[232] 	14
US Billboard Hot 100[233] 	7
US Adult Contemporary (Billboard)[234] 	24
Vietnam (Vietnam Hot 100)[235] 	54
Year-end charts
2024 year-end chart performance for "Bad Blood (Taylor's Version)" Chart (2024) 	Position
US Adult Contemporary (Billboard)[236] 	41
Certifications
Certifications for "Bad Blood (Taylor's Version)" Region 	Certification 	Certified units/sales
Australia (ARIA)[220] 	Gold 	35,000‡
Brazil (Pro-Música Brasil)[53] 	Gold 	20,000‡
United Kingdom (BPI)[237] 	Silver 	200,000‡

‡ Sales+streaming figures based on certification alone.
See also

    List of number-one singles of 2015 (Australia)
    List of number-one singles from the 2010s (New Zealand)
    List of Canadian Hot 100 number-one singles of 2015
    List of Billboard Hot 100 number ones of 2015
    List of number-one digital songs of 2015 (US)
    List of Billboard Mainstream Top 40 number-one songs of 2015
    List of Scottish number-one singles of 2015

Footnotes

Remix only
Attributed to D'Addario,[86] Stereogum's Tom Breihan,[87] and Complex's Constant Gardner,[88]

    Attributed to Time's Daniel D'Addario,[89] Entertainment Weekly's Megan Daley,[90] The A.V. Club's Kayla Kumari Upadhyaya,[82] Slate's Sharan Shetty,[91] The Atlantic's Spencer Kornhaber,[92] and Billboard's Erin Strecker[93]

References

McNutt 2020, p. 78.
Caulfield, Keith (October 30, 2012). "Taylor Swift's Red Sells 1.21 Million; Biggest Sales Week for an Album Since 2002". Billboard. Archived from the original on February 1, 2013. Retrieved February 4, 2019.
McNutt 2020, p. 77.
Light, Alan (December 5, 2014). "Billboard Woman of the Year Taylor Swift on Writing Her Own Rules, Not Becoming a Cliche and the Hurdle of Going Pop". Billboard. Archived from the original on December 26, 2014. Retrieved February 27, 2019.
Talbott, Chris (October 13, 2013). "Taylor Swift Talks Next Album, CMAs and Ed Sheeran". Associated Press. Archived from the original on October 26, 2013. Retrieved October 26, 2013.
Eells, Josh (September 16, 2014). "Taylor Swift Reveals Five Things to Expect on 1989". Rolling Stone. Archived from the original on November 16, 2018. Retrieved November 16, 2018.
Sisario, Ben (November 5, 2014). "Sales of Taylor Swift's 1989 Intensify Streaming Debate". The New York Times. Archived from the original on November 11, 2014. Retrieved February 27, 2019.
Pettifier, Amy (November 27, 2014). "Taylor Swift 1989". The Quietus. Archived from the original on February 7, 2023. Retrieved November 19, 2020.
Perone 2017, p. 55–56.
Dickey, Jack (November 13, 2014). "The Power of Taylor Swift". Time. Archived from the original on August 19, 2020. Retrieved August 5, 2014.
Swift, Taylor (2014). 1989 (Compact disc liner notes). Big Machine Records. BMRBD0500A.
Horton, Matthew (October 27, 2014). "Taylor Swift – 1989". NME. Archived from the original on October 27, 2014. Retrieved July 12, 2024.
Perone 2017, p. 61.
Pareles, Jon (February 11, 2016). "Make Me a Song". The New York Review of Books. Archived from the original on November 27, 2020. Retrieved November 19, 2020.
Geffen, Sasha (October 30, 2014). "Taylor Swift – 1989". Consequence of Sound. Archived from the original on December 28, 2014. Retrieved October 30, 2014.
Caramanica, Jon (October 23, 2014). "A Farewell to Twang". The New York Times. Archived from the original on November 11, 2014. Retrieved July 12, 2024.
Aswad, Jem (October 24, 2014). "Album Review: Taylor Swift's Pop Curveball Pays Off With 1989". Billboard. Archived from the original on February 8, 2017. Retrieved May 20, 2015.
Empire, Kitty (October 26, 2014). "Taylor Swift: 1989 Review – A Bold, Gossipy Confection". The Observer. Archived from the original on October 26, 2014. Retrieved May 20, 2015.
Fusilli, Jim (October 28, 2014). "Don't Like Taylor Swift? Just 'Shake It Off'". The Wall Street Journal. ProQuest 1617521541.
Gardner, Elysa (October 24, 2014). "Swift Moves On, But the Heartache Remains". USA Today. p. D1. ProQuest 1617803681.
Eells, Josh (September 8, 2014). "Cover Story: The Reinvention of Taylor Swift". Rolling Stone. Archived from the original on December 22, 2017. Retrieved July 10, 2024.
Lang, Cady (July 17, 2019). "A Comprehensive Guide to the Taylor Swift-Katy Perry Feud From 2009 to the 'You Need to Calm Down' Happy Meal Reunion". Time. Archived from the original on September 16, 2020. Retrieved October 9, 2020.
"Taylor Swift and Katy Perry: A Timeline of Their Feud". BBC. June 12, 2019. Archived from the original on August 7, 2024. Retrieved July 12, 2024.
Kinane, Ruth (April 5, 2024). "Taylor Swift and Katy Perry: A Timeline of Their Feud". Entertainment Weekly. Archived from the original on July 12, 2024. Retrieved July 12, 2024.
Klosterman, Chuck (October 15, 2015). "Taylor Swift on 'Bad Blood', Kanye West, and How People Interpret Her Lyrics". GQ. Archived from the original on October 18, 2015. Retrieved October 18, 2015.
Trust, Gary (May 21, 2015). "Wiz Khalifa Tops Hot 100, Taylor Swift Re-Enters Following BBMAs Video Premiere". Billboard. Retrieved September 16, 2024.
Hunt, Elle (May 18, 2015). "Taylor Swift Debuts Star-Studded Video for 'Bad Blood' Remix Single". The Guardian. Archived from the original on December 29, 2015. Retrieved November 19, 2020.
"Taylor Swift 'Reached Out' to Kendrick Lamar for 'Bad Blood'". The Washington Post. May 28, 2015. Archived from the original on February 19, 2016. Retrieved July 13, 2024.
Molanphy, Chris (June 3, 2015). "Why Is Taylor Swift and Kendrick Lamar's 'Bad Blood' No. 1?". Slate. Archived from the original on December 22, 2015. Retrieved December 23, 2015.
Britton, Luke (December 17, 2017). "Kendrick Lamar Explains What It Was like Working with Taylor Swift". NME. Archived from the original on October 27, 2020. Retrieved November 10, 2020.
Hiatt, Brian (August 9, 2017). "Kendrick Lamar: The Rolling Stone Interview". Rolling Stone. Archived from the original on November 19, 2020. Retrieved November 19, 2020.
"Bad Blood (feat. Kendrick Lamar) – Single". iTunes Store (US). Archived from the original on May 21, 2015. Retrieved May 18, 2015.
Strecker, Erin (May 17, 2015). "Taylor Swift's 'Bad Blood' Video Premieres". Billboard. Archived from the original on May 21, 2015. Retrieved November 19, 2020.
"Top 40/M Future Releases". All Access. Archived from the original on May 18, 2015.
"Rhythm | Genres". Republic Records. Archived from the original on June 7, 2015. Retrieved July 9, 2015.
Aldi, Giorgia (June 8, 2015). "Taylor Swift – Bad Blood (Radio Date: 12-06-2015)". EarOne. Archived from the original on March 4, 2016. Retrieved June 8, 2015.
Trust, Gary (May 28, 2015). "Taylor Swift's 'Bad Blood' Blasts to No. 1 on Hot 100". Billboard. Archived from the original on September 6, 2015. Retrieved May 28, 2015.
Trust, Gary (July 1, 2015). "Wiz Khalifa No. 1 on Hot 100 'Again,' Selena Gomez Debuts at No. 9". Billboard. Archived from the original on October 2, 2020. Retrieved July 1, 2015.
Trust, Gary (June 22, 2015). "Taylor Swift's 'Bad Blood' Tops Pop Songs Chart". Billboard. Archived from the original on October 18, 2019. Retrieved October 18, 2019.
Trust, Gary (July 13, 2015). "Taylor Swift's 'Bad Blood' Tops Another Tally & Breaks Weekly Plays Record". Billboard. Archived from the original on June 29, 2022. Retrieved June 29, 2022.
Trust, Gary (June 1, 2015). "Chart Highlights: Taylor Swift Ties Record as 'Bad Blood' Hits Pop Songs Top 10". Billboard. Archived from the original on August 7, 2024. Retrieved July 13, 2024.
"2015 Nielsen Music U.S. Report" (PDF). Nielsen SoundScan. January 6, 2016. Archived from the original (PDF) on May 30, 2019. Retrieved January 6, 2016.
"American single certifications – Taylor Swift – Bad Blood". Recording Industry Association of America. Retrieved March 14, 2020.
Trust, Gary (July 14, 2019). "Ask Billboard: Taylor Swift's Career Sales & Streaming Totals, From 'Tim McGraw' to 'You Need to Calm Down'". Billboard. Archived from the original on July 15, 2019. Retrieved July 14, 2019.
"Taylor Swift feat. Kendrick Lamar – Bad Blood". ARIA Top 50 Singles. Retrieved May 30, 2015.
"Taylor Swift Chart History (Canadian Hot 100)". Billboard. Retrieved May 12, 2015.
"Taylor Swift feat. Kendrick Lamar – Bad Blood". Top 40 Singles. Retrieved May 30, 2015.
"Official Scottish Singles Sales Chart Top 100". Official Charts Company. Retrieved May 24, 2015.
"EMA Top 10 Airplay: Week Ending 2015-07-28". Entertainment Monitoring Africa. Retrieved July 30, 2015.
"The official lebanese Top 20 – Taylor Swift". The Official Lebanese Top 20. Archived from the original on September 17, 2016. Retrieved September 1, 2016.
"Official Singles Chart Top 100". Official Charts Company. Retrieved May 24, 2015.
"ARIA Charts – Accreditations – 2023 Singles" (PDF). Australian Recording Industry Association.
"Brazilian single certifications – Taylor Swift – Bad Blood" (in Portuguese). Pro-Música Brasil. Retrieved July 22, 2024.
"Canadian single certifications – Taylor Swift – Bad Blood". Music Canada. Retrieved December 1, 2015.
"British single certifications – Taylor Swift – Bad Blood". British Phonographic Industry. Retrieved September 27, 2024.
"Austrian single certifications – Taylor Swift – Bad Blood" (in German). IFPI Austria. Retrieved May 29, 2024.
"Norwegian single certifications – Taylor Swift – Bad Blood" (in Norwegian). IFPI Norway. Retrieved November 27, 2020.
"Portuguese single certifications – Taylor Swift – Bad Blood" (PDF) (in Portuguese). Associação Fonográfica Portuguesa. Retrieved June 21, 2024.
Cite error: The named reference rmnz was invoked but never defined (see the help page).
White, Jack (July 1, 2021). "Taylor Swift's Top 10 Biggest Collaborations at the Official UK Chart". Official Charts Company. Archived from the original on July 1, 2021. Retrieved July 20, 2023.
Diver, Mike (April 11, 2014). "Taylor Swift – 1989". Clash. Archived from the original on July 13, 2017. Retrieved July 27, 2015.
Lustig, Jay (October 28, 2014). "Taylor Swift's New Album, 1989, Is Pure Pop". The Record. p. BL1. ProQuest 1617292521.
Perone 2017, p. 62.
Unterberger, Andrew (October 28, 2014). "Taylor Swift Gets Clean, Hits Reset on New Album 1989". Spin. Archived from the original on November 19, 2018. Retrieved April 5, 2018.
Wood, Mikael (October 27, 2014). "Taylor Swift Smooths Out the Wrinkles on Sleek 1989". Los Angeles Times. Archived from the original on November 15, 2014. Retrieved February 4, 2019.
Zoladz, Lindsay (October 27, 2014). "Taylor Swift's 1989 Is Her Most Conservative Album Yet". Vulture. Archived from the original on January 10, 2018. Retrieved November 10, 2020.
Sheffield, Rob (April 25, 2024). "All 274 of Taylor Swift's Songs, Ranked". Rolling Stone. Archived from the original on February 15, 2020. Retrieved September 16, 2024.
Song, Jane (February 11, 2020). "All 158 Taylor Swift Songs, Ranked". Paste. Archived from the original on April 13, 2020. Retrieved November 19, 2020.
Markovitz, Adam (November 11, 2014). "1989: Review". Entertainment Weekly. Archived from the original on December 24, 2017. Retrieved July 27, 2015.
Rayner, Ben (October 28, 2014). "Taylor Swift Has Joined Her Pop Peers – Unfortunately". Toronto Star. p. E4. ProQuest 1616974368.
Leedham, Robert (October 30, 2014). "Album Review: Taylor Swift – 1989". Drowned in Sound. Archived from the original on November 25, 2018. Retrieved August 14, 2020.
Sorosiak, Amara (August 25, 2023). "The Scathing Meaning Behind 'Bad Blood' by Taylor Swift". American Songwriter. Archived from the original on July 20, 2024. Retrieved August 7, 2024.
Jones, Nate (May 20, 2024). "All 245 Taylor Swift Songs, Ranked". Vulture. Archived from the original on September 13, 2019. Retrieved September 16, 2024.
Brown, August (May 18, 2015). "Taylor Swift's 'Bad Blood' Video Has Sound, Fury and Kendrick Lamar". Los Angeles Times. Archived from the original on September 2, 2017. Retrieved May 28, 2015.
Petridis, Alexis (April 26, 2019). "Taylor Swift's Singles – Ranked!". The Guardian. Archived from the original on April 27, 2019. Retrieved May 2, 2020.
"Songs of the Year 2015". NME. December 2, 2015. Archived from the original on April 28, 2019. Retrieved December 5, 2015.
"The 90 Best Songs of 2015". PopMatters. January 3, 2016. Archived from the original on January 30, 2016. Retrieved January 3, 2016.
Vena, Jocelyn (May 7, 2015). "Taylor Swift to Debut 'Bad Blood' Video During 2015 Billboard Music Awards". Billboard. Archived from the original on May 10, 2015. Retrieved May 12, 2015.
Levine, Nick (August 21, 2019). "Taylor Swift's Lover: The Struggle to Maintain Superstardom". BBC. Archived from the original on March 1, 2021. Retrieved November 12, 2020.
Ryan, Patrick (September 29, 2015). "Pop Stars, Newcomers Wave a Feminist Flag". USA Today. Archived from the original on August 7, 2024. Retrieved August 1, 2024.
"Watch Taylor Swift's Futuristic, Neo-Noir 'Bad Blood' Video". Rolling Stone. May 17, 2015. Archived from the original on May 18, 2015. Retrieved May 18, 2015.
Upadhyaya, Kayla Kumari (May 18, 2015). "Taylor Swift's New Video Pays Tribute to Every Sci-Fi and Action Film Ever, Basically". The A.V. Club. Archived from the original on August 7, 2024. Retrieved July 29, 2024.
Roth 2018, p. 5.
"30 Taylor Swift Music Videos, Ranked". Spin. November 12, 2017. Archived from the original on November 13, 2017. Retrieved August 1, 2024.
Strecker, Erin (May 21, 2015). "Taylor Swift's 'Bad Blood' Video Breaks Vevo Record". Billboard. Archived from the original on June 23, 2015. Retrieved May 21, 2015.
D'Addario, Daniel (May 18, 2015). "The 'Bad Blood' Music Video Proves Taylor Swift's Power of Reinvention". Time. Archived from the original on August 7, 2024. Retrieved July 30, 2024.
Breihan, Tom (July 15, 2024). "The Number Ones: Taylor Swift's 'Bad Blood' (Feat. Kendrick Lamar)". Stereogum. Archived from the original on July 19, 2024. Retrieved July 30, 2024.
Gardner, Constant (May 18, 2015). "Watch Taylor Swift's Star-Studded 'Bad Blood' Video Co-Starring Kendrick Lamar". Cosmopolitan. Archived from the original on July 30, 2024. Retrieved July 30, 2024.
D'Addario, Daniel (May 18, 2015). "Watch Taylor Swift's Star-Studded Music Video for 'Bad Blood'". Time. Archived from the original on May 18, 2017. Retrieved May 18, 2015.
Daley, Megan (May 18, 2015). "Fights, Fire and Revenge: The Best of Taylor Swift's 'Bad Blood'". Entertainment Weekly. Archived from the original on August 7, 2024. Retrieved July 29, 2024.
Shetty, Sharan (May 18, 2015). "Taylor Swift's 'Bad Blood' Video Has Enough Celebrities to Be an Actual Movie". Slate. Archived from the original on July 26, 2020. Retrieved May 17, 2015.
Kornhaber, Spencer (May 18, 2015). "Taylor Swift's 'Bad Blood' Video Is the Anti-Avengers". The Atlantic. Archived from the original on November 25, 2020. Retrieved December 1, 2020.
Strecker, Erin (May 17, 2015). "Taylor Swift's 'Bad Blood' Video: 15 Things We Need To Talk About Right Now". Billboard. Archived from the original on January 25, 2021. Retrieved May 18, 2015.
Miller, Matt (October 27, 2017). "Sci-Fi Taylor Swift Is the Best Taylor Swift". Esquire. Archived from the original on July 29, 2024. Retrieved July 29, 2024.
Siroky, Mary (November 9, 2021). "Every Taylor Swift Album Ranked from Worst to Best". Consequence. Archived from the original on March 28, 2022. Retrieved November 10, 2021.
Yahr, Emily (May 17, 2015). "What Taylor Swift's 'Bad Blood' Music Video Actually Says About Power in Hollywood". The Washington Post. Retrieved August 1, 2024.
Gannon, Jennifer (June 9, 2016). "Taylor Swift: Why Is It So Difficult to Support Her?". The Irish Times. Archived from the original on January 7, 2021. Retrieved December 1, 2020.
Barlow, Eve (February 14, 2016). "Taylor Swift: She's the Boss". The Times. Archived from the original on June 13, 2024. Retrieved August 12, 2024.
Isaksen & Eltantawy 2019, p. 559.
Roth 2018, p. 4.
Lipshutz, Jason (August 31, 2015). "MTV Video Music Awards 2015: The Winners Are..." Billboard. Archived from the original on February 25, 2022. Retrieved September 5, 2024.
Medved, Matt (August 31, 2015). "Taylor Swift Wins First Video of the Year VMA for 'Bad Blood'". Billboard. Archived from the original on July 16, 2024. Retrieved August 1, 2024.
"2016 Grammy Awards: Complete List of Winners and Nominees". Los Angeles Times. December 7, 2015. Archived from the original on January 3, 2016. Retrieved September 5, 2024.
"2016 ASCAP Pop Music Awards". American Society of Composers, Authors and Publishers. April 28, 2016. Archived from the original on May 2, 2016. Retrieved April 28, 2016.
"BMI Honors Taylor Swift and Legendary Songwriting Duo Mann & Weil at the 64th Annual BMI Pop Awards". Broadcast Music, Inc. May 11, 2016. Archived from the original on June 2, 2016. Retrieved August 1, 2024.
"Winners of Teen Choice 2015 Announced". Teen Choice Awards. August 16, 2015. Archived from the original on April 14, 2020. Retrieved August 17, 2015.
Vivarelli, Nick (October 25, 2015). "Justin Bieber Dominates at MTV Europe Music Awards". Variety. Archived from the original on August 1, 2024. Retrieved August 1, 2024.
Iasimone, Ashley (May 2, 2016). "2016 Radio Disney Music Awards: See the Full List of Winners". Billboard. Retrieved August 1, 2024.
Blistein, Jon (October 13, 2015). "Taylor Swift, Ed Sheeran, the Weeknd Top AMA Nominations". Rolling Stone. Archived from the original on January 31, 2017. Retrieved May 23, 2017.
"Nominees & Winners". People's Choice Awards. Archived from the original on November 4, 2015. Retrieved November 3, 2015.
Grant, Stacey (February 2, 2016). "Here Are The Nominees For The 2016 Kids' Choice Awards". MTV. Archived from the original on February 4, 2016. Retrieved February 3, 2016.
Lynch, Joe (February 9, 2016). "iHeartRadio Music Awards Announce 2016 Nominees, Performers & New Categories". Billboard. Archived from the original on February 9, 2016. Retrieved February 9, 2016.
"Brillan hasta en ausencia" [They Shine Even in Absence]. El Universal (in Spanish). November 27, 2015. Archived from the original on August 7, 2024. Retrieved August 1, 2024.
"MYXMusicAwards 2016 Winners List". Myx. March 16, 2016. Archived from the original on August 10, 2016. Retrieved August 6, 2016.
"Taylor Swift : grande gagnante des NRJ Music Awards 2015 !" [Taylor Swift: Big Winner of the 2015 NRJ Music Awards!] (in French). NRJ. November 8, 2015. Retrieved August 1, 2024.
"Taylor Swift & Nicki Minaj Declare No 'Bad Blood' With Joint Performance: Watch". Billboard. August 30, 2015. Archived from the original on January 31, 2021. Retrieved November 19, 2020.
Hall, David Brendan (October 23, 2016). "Taylor Swift Delivers a Knockout Performance at Formula 1 Concert in Austin". Billboard. Archived from the original on September 2, 2022. Retrieved July 19, 2024.
Atkinson, Katie (February 5, 2017). "Taylor Swift Performs 'Better Man' & 'I Don't Wanna Live Forever' for First Time at Stunning Pre-Super Bowl Set". Billboard. Archived from the original on May 25, 2022. Retrieved July 19, 2024.
"Taylor Swift Comes 'Home' to Philly, Shows There's No Going Back with Her Music". The Morning Call. June 14, 2015. Archived from the original on January 17, 2024. Retrieved July 19, 2024.
Sheffield, Rob (May 9, 2018). "Why Taylor Swift's 'Reputation' Tour Is Her Finest Yet". Rolling Stone. Archived from the original on May 10, 2018. Retrieved May 10, 2018.
O'Connor, Roisin (June 8, 2018). "Taylor Swift 'Reputation' Stadium Tour Review: Dazzling Pop Spectacle from the Star Who Doesn't Stand Still". The Independent. Archived from the original on May 26, 2022. Retrieved July 19, 2024.
Princiotti, Nora (March 16, 2023). "On the Eve of Eras, Ranking Taylor Swift's All-Time Best Live Performances". The Ringer. Archived from the original on March 31, 2023. Retrieved August 2, 2024.
Savage, Mark (June 8, 2024). "Taylor Swift Eras Tour Review: Pop's Heartbreak Princess Dazzles in Edinburgh". BBC. Archived from the original on August 7, 2024. Retrieved July 19, 2024.
Raggio, Eva (April 1, 2023). "Taylor Swift Kicked Off Her Arlington Concerts Like the Icon She Is, No Matter Who Disagrees". Dallas Observer. Archived from the original on April 2, 2023. Retrieved August 2, 2024.
Strecker, Erin (August 12, 2015). "This Taylor Swift 'Bad Blood' Parody Video Targets U.K. Tampon Tax". Billboard. Retrieved September 6, 2024.
Robinson, Will (September 10, 2015). "Taylor Swift's 'Bad Blood' Meets Batman v Superman: Dawn of Justice". Entertainment Weekly. Archived from the original on January 26, 2021. Retrieved November 10, 2020.
Waxman, Olivia B. (September 15, 2015). "This 'Bad Blood' Parody Is Raising Awareness for Bone Marrow Donations". Time. Retrieved September 6, 2024.
Armstrong, Megan (October 4, 2017). "Nicole Richie & Tina Fey Spoof Taylor Swift's 'Bad Blood' Video on Great News: Watch". Billboard. Archived from the original on September 30, 2022. Retrieved September 6, 2024.
Renshaw, David (June 24, 2015). "Drenge Cover Taylor Swift's 'Bad Blood' – Listen". NME. Archived from the original on May 12, 2021. Retrieved November 19, 2020.
Stutz, Colin (July 9, 2015). "Alessia Cara Gets Mad Love From Taylor Swift for 'Bad Blood' Cover: Watch". Billboard. Archived from the original on August 1, 2024. Retrieved August 1, 2024.
Lipshutz, Jason (November 20, 2016). "Watch Drake Jam To Taylor Swift's 'Bad Blood' in New Apple Music Ad". Billboard. Archived from the original on February 1, 2021. Retrieved November 19, 2020.
Kreps, Daniel (July 6, 2015). "Hear Taylor Swift's 'Bad Blood' Sung in 20 Different Styles". Rolling Stone. Archived from the original on August 16, 2022. Retrieved September 9, 2024.
Anderson, L. V. (September 17, 2015). "Ryan Adams Shares 'Bad Blood', the First Full Track From His Taylor Swift Cover Album". Slate. Retrieved September 16, 2024.
Browne, David (September 21, 2015). "Ryan Adams on His Full-Album Cover of Taylor Swift's 1989". Rolling Stone. Archived from the original on September 25, 2023. Retrieved December 29, 2023.
Spanos, Brittany (September 17, 2015). "Hear Ryan Adams' Moody Cover of Taylor Swift's 'Bad Blood'". Rolling Stone. Archived from the original on January 17, 2021. Retrieved November 19, 2020.
Murphy, Sarah (September 22, 2015). "Ryan Adams: 1989". Exclaim!. Archived from the original on November 3, 2023. Retrieved August 2, 2024.
Wood, Mikael (September 21, 2015). "Review: Ryan Adams Turns to Taylor Swift for Help On His Version of 1989". Los Angeles Times. Archived from the original on September 23, 2015. Retrieved September 21, 2015.
Breihan, Tom (September 17, 2015). "Ryan Adams – 'Bad Blood' (Taylor Swift Cover)". Stereogum. Archived from the original on August 1, 2024. Retrieved August 1, 2024.
Unterberger, Andrew (September 23, 2015). "Ryan Adams' 1989: A Worthwhile Disappointment". Spin. Archived from the original on November 27, 2020. Retrieved November 19, 2020.
Zaleski, Annie (September 21, 2015). "Ryan Adams Transforms Taylor Swift's 1989 Into A Melancholy Masterpiece". The A.V. Club. Archived from the original on February 12, 2018. Retrieved February 12, 2018.
Payne, Chris (September 21, 2015). "We Ranked All of Ryan Adams' Covers of Taylor Swift's 1989". Billboard. Archived from the original on August 2, 2024. Retrieved August 2, 2024.
Mapes, Jillian (September 22, 2015). "Ryan Adams's 1989 Is a Decent Breakup Album, Not a Poptimist Manifesto". Vulture. Archived from the original on August 2, 2024. Retrieved August 2, 2024.
"Bad Blood" (in French). Ultratop. Hung Medien. Archived from the original on August 1, 2020. Retrieved November 19, 2020.
"Ryan Adams Chart History". Billboard. Archived from the original on November 30, 2022. Retrieved November 19, 2020.
"'Bad Blood' by Taylor Swift / Kendrick Lamar". Tidal. Retrieved September 9, 2024.
"Taylor Swift feat. Kendrick Lamar – Bad Blood" (in German). Ö3 Austria Top 40. Retrieved May 30, 2015.
"Taylor Swift feat. Kendrick Lamar – Bad Blood" (in Dutch). Ultratop 50. Retrieved May 30, 2015.
"Taylor Swift feat. Kendrick Lamar – Bad Blood" (in French). Ultratop 50. Retrieved May 30, 2015.
"Top 100 Billboard Brasil". Billboard Brasil. July 6, 2015. Archived from the original on July 13, 2015. Retrieved July 12, 2015.
"Taylor Swift Chart History (Canada AC)". Billboard. Retrieved July 18, 2015.
"Taylor Swift Chart History (Canada CHR/Top 40)". Billboard. Retrieved July 11, 2015.
"Taylor Swift Chart History (Canada Hot AC)". Billboard. Retrieved July 18, 2015.
"ČNS IFPI" (in Czech). Hitparáda – Radio Top 100 Oficiální. IFPI Czech Republic. Note: Select 23. týden 2015 in the date selector. Retrieved May 30, 2015.
"Taylor Swift Chart History (Euro Digital Song Sales)". Billboard. Retrieved May 28, 2015.
"Taylor Swift: Bad Blood (Feat. Kendrick Lamar)" (in Finnish). Musiikkituottajat. Retrieved June 4, 2015.
"Taylor Swift – Bad Blood" (in French). Les classement single. Retrieved May 30, 2015.
"Taylor Swift feat. Kendrick Lamar – Bad Blood" (in German). GfK Entertainment charts. Retrieved May 30, 2015.
"Taylor Swift Chart History (Greece Digital Song Sales)". Billboard. Archived from the original on December 6, 2019. Retrieved November 9, 2021.
"Archívum – Slágerlisták – MAHASZ" (in Hungarian). Single (track) Top 40 lista. Magyar Hanglemezkiadók Szövetsége. Retrieved May 28, 2015.
"Chart Track: Week 22, 2015". Irish Singles Chart. Retrieved May 19, 2015.
"Top Digital – Classifica settimanale WK 21 (dal 2015-05-18 al 2015-05-24)". Federazione Industria Musicale Italiana. Archived from the original on March 30, 2017. Retrieved September 25, 2015.
"Taylor Swift Chart History (Japan Hot 100)". Billboard. Retrieved June 15, 2015.
"Japan Adult Contemporary Airplay Chart". Billboard Japan (in Japanese). Archived from the original on October 31, 2023. Retrieved October 31, 2023.
"Taylor Swift Chart History (Mexico Airplay)". Billboard. Archived from the original on August 19, 2019. Retrieved February 22, 2021.
"Nederlandse Top 40 – Taylor Swift feat. Kendrick Lamar" (in Dutch). Dutch Top 40. Retrieved May 30, 2015.
"Taylor Swift – Bad Blood". Top 40 Singles. Retrieved September 30, 2021.
"ČNS IFPI" (in Slovak). Hitparáda – Radio Top 100 Oficiálna. IFPI Czech Republic. Note: insert 201523 into search. Retrieved May 30, 2015.
"Taylor Swift feat. Kendrick Lamar – Bad Blood". Swiss Singles Chart. Retrieved May 24, 2015.
"Taylor Swift Chart History (Hot 100)". Billboard. Retrieved May 27, 2015.
"Taylor Swift Chart History (Adult Contemporary)". Billboard. Retrieved August 25, 2015.
"Taylor Swift Chart History (Adult Pop Songs)". Billboard. Retrieved July 14, 2015.
"Taylor Swift Chart History (Dance Mix/Show Airplay)". Billboard. Retrieved July 2, 2015.
"Taylor Swift Chart History (Dance Club Songs)". Billboard. Retrieved July 21, 2015.
"Taylor Swift Chart History (Pop Songs)". Billboard. Retrieved June 25, 2015.
"Taylor Swift Chart History (Rhythmic)". Billboard. Retrieved August 4, 2015.
"Record Report – Rock General" (in Spanish). Record Report. Archived from the original on July 14, 2015. Retrieved July 14, 2015.
"Taylor Swift Chart History (Global 200)". Billboard. Retrieved August 22, 2023.
"Taylor Swift feat. Kendrick Lamar – Bad Blood". AFP Top 100 Singles. Retrieved November 9, 2023.
"RIAS Top Charts Week 44 (27 Oct – 2 Nov 2023)". RIAS. Archived from the original on November 7, 2023. Retrieved November 7, 2023.
"Top 100 Singles 2015". Australian Recording Industry Association. Archived from the original on January 24, 2016. Retrieved January 6, 2016.
"Canadian Hot 100 Year End 2015". Billboard. Archived from the original on November 18, 2016. Retrieved December 9, 2015.
"Single Top 100 – eladási darabszám alapján – 2015" (in Hungarian). Mahasz. Archived from the original on March 5, 2016. Retrieved January 11, 2021.
"End of Year Singles Chart Top 100 – 2015". Official Charts Company. Archived from the original on February 12, 2016. Retrieved January 5, 2016.
"Hot 100: Year End 2015". Billboard. Archived from the original on December 22, 2015. Retrieved December 9, 2015.
"Adult Contemporary Songs Year End 2015". Billboard. Archived from the original on December 11, 2015. Retrieved December 9, 2015.
"Adult Pop Songs Year End 2015". Billboard. Archived from the original on December 14, 2015. Retrieved December 9, 2015.
"Dance/Mix Show Songs Year End 2015". Billboard. Archived from the original on August 2, 2016. Retrieved August 25, 2016.
"Pop Songs Year End 2015". Billboard. Archived from the original on November 18, 2016. Retrieved December 9, 2015.
"Rhythmic Songs Year End 2015". Billboard. Archived from the original on August 2, 2016. Retrieved August 25, 2016.
"As 100 Mais Tocadas nas Rádios Jovens em 2016". Billboard Brasil (in Portuguese). January 4, 2017. Archived from the original on September 7, 2017. Retrieved September 7, 2017.
"Danish single certifications – Taylor Swift – Bad Blood". IFPI Danmark. Retrieved July 5, 2023.
"Gold-/Platin-Datenbank (Taylor Swift feat. Kendrick Lamar; 'Bad Blood')" (in German). Bundesverband Musikindustrie. Retrieved June 4, 2023.
"Italian single certifications – Taylor Swift – Bad Blood" (in Italian). Federazione Industria Musicale Italiana. Retrieved August 7, 2023.
"New Zealand single certifications – Taylor Swift feat. Kendrick Lamar – Bad Blood". Radioscope. Retrieved December 19, 2024. Type Bad Blood in the "Search:" field.
"Spanish single certifications – Taylor Swift / Kendrick Lamar – Bad Blood". El portal de Música. Productores de Música de España. Retrieved July 23, 2024.
"'Bad Blood (feat. Kendrick Lamar)' – (Radio Date: 12/06/2015)". radioairplay.fm. Archived from the original on June 16, 2015. Retrieved June 8, 2015.
Melas, Chloe (November 16, 2020). "Taylor Swift Speaks Out About Sale of Her Masters". CNN. Archived from the original on November 18, 2020. Retrieved November 19, 2020.
"Taylor Swift Wants to Re-Record Her Old Hits". BBC News. August 22, 2019. Archived from the original on August 22, 2019. Retrieved August 22, 2019.
Finnis, Alex (November 17, 2020). "Taylor Swift Masters: The Controversy around Scooter Braun Selling the Rights to Her Old Music Explained". i. Archived from the original on February 12, 2021. Retrieved February 13, 2021.
Shah, Neil (April 9, 2021). "Taylor Swift Releases New Fearless Album, Reclaiming Her Back Catalog". The Wall Street Journal. Archived from the original on October 8, 2021. Retrieved September 25, 2022.
Espada, Mariah (July 6, 2023). "Taylor Swift Is Halfway Through Her Rerecording Project. It's Paid Off Big Time". Time. Archived from the original on October 27, 2023. Retrieved November 6, 2023.
Uitti, Jacob (July 30, 2022). "The Rock Teases Unreleased 'Bad Blood (Taylor's Version)' on TikTok, Swift Approves". American Songwriter. Archived from the original on August 5, 2022. Retrieved August 2, 2022.
Knopper, Steve (August 17, 2022). "How a Kid Flick Got Taylor Swift to Remake a Previously Off-Limits Song". Billboard. Archived from the original on August 18, 2022. Retrieved August 12, 2024.
Vassell, Nicole (October 27, 2023). "Taylor Swift Fans Celebrate As Pop Star Releases 1989 (Taylor's Version)". The Independent. Archived from the original on October 30, 2023. Retrieved December 29, 2023.
Shafer, Ellise (October 27, 2023). "Taylor Swift Thanks Kendrick Lamar for Re-Recording 'Bad Blood' Verse on 1989 (Taylor's Version): 'Surreal and Bewildering'". Variety. Archived from the original on October 27, 2023. Retrieved October 27, 2023.
Aroesti, Rachel (October 27, 2023). "Taylor Swift: 1989 (Taylor's Version) Review – Subtle Bonus Tracks Add New Depths to a Classic". The Guardian. Archived from the original on October 27, 2023. Retrieved August 2, 2024.
D'Souza, Shaad (October 30, 2023). "Taylor Swift: 1989 (Taylor's Version) Album Review". Pitchfork. Archived from the original on October 30, 2023. Retrieved October 30, 2023.
Swift, Taylor (2023). 1989 (Taylor's Version) (Compact disc liner notes). Republic Records. 0245597656.
Braaten, Elizabeth (October 30, 2023). "Taylor Swift: 1989 (Taylor's Version) Album Review". Paste. Archived from the original on November 2, 2023. Retrieved August 2, 2024.
Martin, Rachel (October 27, 2023). "Album Review: 1989 (Taylor's Version) by Taylor Swift". Notion. Archived from the original on May 25, 2024. Retrieved August 2, 2024.
Zylstra, Tione (October 28, 2023). "Album Review: Taylor Swift – 1989 (Taylor's Version)". The Music. Archived from the original on August 2, 2024. Retrieved August 2, 2024.
Power, Ed (October 27, 2023). "Taylor Swift's 1989 (Taylor's Version) Is Still Thrilling – No Wonder It Went Supernova". i. Archived from the original on October 27, 2023. Retrieved August 2, 2024.
Sutherland, Mark (October 27, 2023). "Taylor Swift, 1989 (Taylor's Version) Could Be the Best Pop Album of 2023". Rolling Stone UK. Archived from the original on October 29, 2023. Retrieved August 2, 2024.
Trust, Gary (November 6, 2023). "Taylor Swift's 'Is It Over Now? (Taylor's Version)' Debuts at No. 1 on Billboard Hot 100". Billboard. Archived from the original on November 6, 2023. Retrieved November 7, 2023.
Zellner, Xander (November 6, 2023). "Taylor Swift Charts All 21 Songs From 1989 (Taylor's Version) on the Hot 100". Billboard. Archived from the original on November 6, 2023. Retrieved November 7, 2023.
Trust, Gary (November 6, 2023). "Taylor Swift Makes History With Top 6 Songs, All From 1989 (Taylor's Version), on Billboard Global 200 Chart". Billboard. Archived from the original on November 14, 2023. Retrieved November 7, 2023.
"Taylor Swift Chart History (Canadian Hot 100)". Billboard. Retrieved November 7, 2023.
"Taylor Swift – Bad Blood (Taylor's Version)". Top 40 Singles. Retrieved August 2, 2024.
"Taylor Swift Chart History (Philippines Songs)". Billboard. Archived from the original on November 8, 2023. Retrieved November 7, 2023.
"ARIA Charts – Accreditations – 2024 Singles" (PDF). Australian Recording Industry Association. Retrieved June 14, 2024.
"The ARIA Report: Week Commencing 6 November 2023". The ARIA Report. No. 1757. Australian Recording Industry Association. November 6, 2023. p. 4.
"Taylor Swift Chart History (Brasil Hot 100)". Billboard. Retrieved November 9, 2023.
"Taylor Swift Chart History (Global 200)". Billboard. Retrieved November 7, 2023.
"Digital Singles Chart (International)". IFPI Greece. Archived from the original on November 13, 2023. Retrieved November 8, 2023.
"Taylor Swift Chart History (Ireland Songs)". Billboard. Archived from the original on November 7, 2023. Retrieved November 7, 2023.
"TOP 20 Most Streamed International Singles In Malaysia Week 44 (27/10/2023- 02/11/2023)". RIM. November 11, 2023. Archived from the original on November 12, 2023. Retrieved November 12, 2023 – via Facebook.
"Taylor Swift – Bad Blood (Taylor's Version)". Singles Top 100. Retrieved August 2, 2024.
"This Week's Official UAE Chart Top 20: from 27/10/2023 to 02/11/2023". International Federation of the Phonographic Industry. October 27, 2023. Archived from the original on November 8, 2023. Retrieved November 8, 2023.
"Taylor Swift Chart History (U.K. Songs)". Billboard. Archived from the original on November 7, 2023. Retrieved November 7, 2023.
"Official Singles Downloads Chart Top 100". Official Charts Company. Retrieved November 3, 2023.
"Official Singles Sales Chart Top 100". Official Charts Company. Archived from the original on November 11, 2023. Retrieved November 3, 2023.
"Official Streaming Chart Top 100". Official Charts Company. Archived from the original on November 3, 2023. Retrieved November 3, 2023.
"Taylor Swift Chart History (Hot 100)". Billboard. Retrieved November 6, 2023.
"Taylor Swift Chart History (Adult Contemporary)". Billboard. Retrieved September 9, 2024.
"Taylor Swift Chart History (Billboard Vietnam Hot 100)". Billboard. Archived from the original on June 6, 2022. Retrieved November 11, 2023.
"Adult Contemporary Songs – Year-End 2024". Billboard. Retrieved December 15, 2024.

    "British single certifications – Taylor Swift – Bad Blood (Taylor's Version)". British Phonographic Industry. Retrieved December 6, 2024.

Sources

    Isaksen, Judy L.; Eltantawy, Nahed (October 25, 2019). "What happens when a celebrity feminist slings microaggressive shade?: Twitter and the pushback against neoliberal feminism". Celebrity Studies. 12 (4): 549–564. doi:10.1080/19392397.2019.1678229.
    McNutt, Myles (2020). "From 'Mine' to 'Ours': Gendered Hierarchies of Authorship and the Limits of Taylor Swift's Paratextual Feminism". Communication, Culture and Critique. 13 (1): 72–91. doi:10.1093/ccc/tcz042.
    Perone, James E. (2017). "1989 and Beyond". The Words and Music of Taylor Swift. ABC-Clio. pp. 55–68. ISBN 9781440852954.
    Roth, Hannelore (2018). "The Feminist Manifesto by Taylor Swift: Boss Babes, Fit Girls and Welvin Da Great". Collateral. Online Journal for Cross-Cultural Close Reading. 14: 1–7. ISSN 2506-7982.

    vte

Taylor Swift songs

    vte

Kendrick Lamar songs

    vte

Ryan Adams
Awards for "Bad Blood"
Authority control databases Edit this at Wikidata	

    MusicBrainz release group
        2

Categories:

    2015 songs2015 singlesBillboard Hot 100 number-one singlesCanadian Hot 100 number-one singlesKendrick Lamar songsMTV Video of the Year AwardGrammy Award for Best Short Form Music VideoMusic videos directed by Joseph KahnRyan Adams songsSong recordings produced by Max MartinSong recordings produced by Shellback (record producer)Song recordings produced by Ilya SalmanzadehSong recordings produced by Taylor SwiftSong recordings produced by Chris RoweSongs written by Taylor SwiftSongs written by Kendrick LamarSongs written by Max MartinSongs written by Shellback (record producer)Taylor Swift songsNumber-one singles in AustraliaNumber-one singles in New ZealandNumber-one singles in ScotlandBig Machine Records singlesSongs containing the I–V-vi-IV progressionAmerican pop songs

    This page was last edited on 30 December 2024, at 21:42 (UTC).
    Text is available under the Creative Commons Attribution-ShareAlike 4.0 License; additional terms may apply. By using this site, you agree to the Terms of Use and Privacy Policy. Wikipedia® is a registered trademark of the Wikimedia Foundation, Inc., a non-profit organization.

    Privacy policy
    About Wikipedia
    Disclaimers
    Contact Wikipedia
    Code of Conduct
    Developers
    Statistics
    Cookie statement
    Mobile view

    Wikimedia Foundation
    Powered by MediaWiki



Wikipedia The Free Encyclopedia

    Donate
    Create account
    Log in

Contents
(Top)
Background and release

Composition and production

Lyrics and interpretations

Critical reception

Commercial performance

Music video

    Production and release
    Synopsis
    Analysis

Live performances and other uses

    Jack Leopards & the Dolphin Club cover
    "Taylor's Version" re-recording

Accolades

Credits and personnel

Charts

    Weekly charts
    Year-end charts

Certifications

Release history

See also

References

        Cited literature

Look What You Made Me Do

    Article
    Talk

    Read
    Edit
    View history

Tools

Appearance
Text

    Small
    Standard
    Large

Width

    Standard
    Wide

Color (beta)

    Automatic
    Light
    Dark

From Wikipedia, the free encyclopedia
For other uses, see Look What You Made Me Do (disambiguation).
"Look What You Made Me Do"
Cover art of "Look What You Made Me Do", a black-and-white photo of Taylor Swift covering her face with her hands, leaving the eyes staring
Single by Taylor Swift
from the album Reputation
Released	August 24, 2017
Studio	Rough Customers (Brooklyn)
Genre	

    Electropop dance-pop progressive pop synth-punk

Length	3:31
Label	Big Machine
Songwriter(s)	

    Taylor Swift Jack Antonoff Fred Fairbrass Richard Fairbrass Rob Manzoli

Producer(s)	

    Taylor Swift Jack Antonoff

Taylor Swift singles chronology
"I Don't Wanna Live Forever"
(2016) 	"Look What You Made Me Do"
(2017) 	"...Ready for It?"
(2017)
Music video
"Look What You Made Me Do" on YouTube

"Look What You Made Me Do" is a song by the American singer-songwriter Taylor Swift and the lead single from her sixth studio album, Reputation (2017). Big Machine Records released the song on August 24, 2017, after approximately one year of Swift's hiatus due to the controversies that affected her "America's Sweetheart" public image throughout 2016. While secluding from public appearances, she wrote and produced the track with Jack Antonoff.

"Look What You Made Me Do" has an electronic production combining electropop, dance-pop, progressive pop, and synth-punk with elements of hip hop, electroclash, industrial, and electro. It contains an interpolation of "I'm Too Sexy" (1991) by the English pop group Right Said Fred, whose members received songwriting credits as a result. The melody incorporates strings, plinking piano, and synthesizers, and the chorus consists of drumbeats and rhythmic chants. The lyrics are about the narrator's contempt for somebody who had wronged them; many media publications interpreted the track to be a reference to the controversies that Swift faced.

The accompanying music video premiered at the 2017 MTV Video Music Awards and contains various implications of Swift's celebrity that received widespread media speculation. Both the song and the video broke streaming records on Spotify and YouTube upon release. "Look What You Made Me Do" polarized music critics: some deemed it a fierce return and an interesting direction but others criticized the sound and theme as harsh and vindictive that strayed away from Swift's singer-songwriter artistry. Critics have considered "Look What You Made Me Do" a career-defining single for Swift.

In the United States, the single peaked atop the Billboard Hot 100 with the highest sales week of 2017 and was certified four-times platinum by the Recording Industry Association of America. The single also peaked atop the singles charts of countries including Australia, Canada, Ireland, the Philippines, and the United Kingdom, and it received multi-platinum certifications in Australia, Brazil, Canada, Poland, Sweden, and the United Kingdom. Swift performed the song on the Reputation Stadium Tour (2018) and the Eras Tour (2023–2024).
Background and release

Taylor Swift released her fifth studio album, 1989, on October 27, 2014,[1] it sold 10 million copies worldwide,[2] and three of the album's singles reached number one on the US Billboard Hot 100.[3] The album propelled Swift to pop stardom;[1] Billboard wrote that it brought forth "a kind of cultural omnipresence that's rare for a 2010s album".[3] Swift's popularity turned her into a media fixation,[4] and her once-wholesome "America's Sweetheart" reputation was tarnished by short-lived romantic relationships and public celebrity controversies, including a feud with rapper Kanye West and his then-wife, media personality Kim Kardashian, over West's song "Famous" (2016), in which he claims he made Swift a success ("I made that bitch famous").[5][6][7]

Although Swift said she never consented to the said lyric, Kardashian released a phone recording between Swift and West, in which the former seemingly consented to another portion of the song.[8] After the West–Kardashian controversy, Swift became a subject of an online "#IsOverParty" hashtag, where online audiences used the "snake" emoji to describe her.[7][9] Detractors regarded Swift as fake and calculating, a conclusion that surmounted after years of what they saw as a deliberate maneuver to carefully cultivate her public image.[10][11] Swift became increasingly reticent on social media despite a large following and avoided the press amidst the commotion and ultimately withdrew from public appearances.[12][13] She went through a hiatus mid-2016 and felt "people might need a break from [her]".[14]

On August 18, 2017, Swift blanked out all of her social media accounts,[15] which prompted media speculation on new music.[16] In the following days, she uploaded silent, black-and-white short videos of CGI snakes onto social media, which attracted widespread press attention.[16][17][18] Imagery of snakes was inspired by the West–Kardashian controversy and featured prominently in Swift's social media posts.[19] On August 23, she announced on Instagram a new album, titled Reputation.[20] The following day, she unveiled the lead single from the album, "Look What You Made Me Do",[21][22] which was released for streaming and download on digital platforms by Big Machine Records.[23] A lyric video was released simultaneously; it was produced by Swift and Joseph Kahn and directed by ODD.[24] The lyric video features prominent snake imagery, depicting the chorus with an ouroboros,[25] and its graphics were influenced by the aesthetics of Saul Bass for the 1958 film Vertigo.[26] It was viewed over 19 million times within the first 24 hours on YouTube, setting a record for the most-viewed lyric video in that time frame.[27]

"Look What You Made Me Do" impacted US contemporary hit radio on August 29, 2017.[28] In Germany, the track was released as a CD single by Universal Music on October 27, 2017.[29] Media publications regarded "Look What You Made Me Do" as Swift's comeback after her year of hiatus from the public spotlight.[30][31][32][33]
Composition and production

Swift wrote and produced "Look What You Made Me Do" with Jack Antonoff, who also programmed the track and played its instruments, recorded at Rough Customer Studio in Brooklyn.[34] Other musicians on the track included Evan Smith (saxophone), Victoria Parker (violin), and Phillip A. Peterson (cello). Laura Sisk engineered the song, and Serban Ghenea mixed it at MixStar Studios in Virginia Beach, Virginia. The track was mastered by Randy Merrill at Sterling Sound in New York.[35]
"Look What You Made Me Do"
Duration: 20 seconds.0:20
A sample of the pre-chorus and chorus of "Look What You Made Me Do"; the pre-chorus incorporates piano and synth-stimulated brass and the chorus consists of drumbeats and rhythmic chants of the song title
Problems playing this file? See media help.

"Look What You Made Me Do" is 3 minutes and 31 seconds long.[23] It is written in the key of A minor and has a tempo of 128 beats per minute.[36] The track begins with string swell and plinking piano keys and progresses into an electronic production; The New York Times wrote that the opening strings and piano were melodramatic and evoked a "dark, fantasy-film" atmosphere,[17] whereas The Daily Telegraph's Sarah Carson described the strings as "Hollywood"–inspired.[37]

The verses and chorus are built on a minor-key motif,[38] electronic tones, and hip-hop–inspired beats and vocal cadences.[17][39] The pre-chorus incorporates piano and synth–simulated brass,[17] and the bridge incorporates strings that swell over a reverberating crescendo.[37] The chorus consists of drumbeats and rhythmic chants of the title, which is repeated eight times with different tones of delivery.[17][38] The melody contains an interpolation of "I'm Too Sexy" (1991) by the English pop group Right Said Fred, leading to its members Richard Fairbrass, Fred Fairbrass, and Rob Manzoli receiving writing credits.[40][41] The Fairbrass brothers were contacted one week before the release of "Look What You Made Me Do" and were asked whether a "big, contemporary female artist who hasn't released anything for a while" would be able to use a portion of their song for her latest single; they found out that the artist in question was Swift after the song was released.[42]

Jon Pareles commented that the piano and strings in the pre-chorus and bridge gave them a "melodramatic, emotional" feel, whereas Swift's repeating the title in the chorus sounded "vindictive, mocking, dismissive, even a little playful".[17] Music critics mostly described the track as electropop.[26][43][44][45] NPR's Lars Gotrich said that the beats and vocals evoked electroclash,[45] Rolling Stone's Brittany Spanos said it was a dance-pop song,[46] and Fact's Chal Ravens deemed it progressive pop.[47] Annie Zaleski characterized the genre as synth-punk.[48] The production incorporates elements of mid-1980s and 1990s industrial and electro.[17][34] Some critics commented that the track showcased a "darker" soundscape compared to Swift's previous releases;[17][34] Spanos attributed this effect partly to the "dark techno" of Britney Spears's 2007 album Blackout,[49] while Maura Johnston in The Guardian said that the atmosphere evoked Spears's "Piece of Me" (2007) and the beat was reminiscent of Peaches's "Operate" (2003).[26]
Lyrics and interpretations

With the key themes being betrayal and vengeance,[17][38] "Look What You Made Me Do" was developed by Swift as a poem about her feelings and realizations that she could not trust certain people and could only rely on a few.[50] The verses contain lyrics such as, "I don't like your perfect crime/ How you laugh when you lie"; according to Pereles, these lyrics resemble a nursery rhyme, and they could be interpreted as referring to either the celebrity feuds or the feelings after a romantic breakup.[17] Swift's character resents that her enemies set her up for humiliation ("You said the gun was mine/ Isn't cool, no, I don't like you"), affirms that she remembers all the wrongdoings ("I got a list of names and yours is in red, underlined/ I check it once, then I check it twice"),[38] and reassures that her enemies will get what they deserve ("Maybe I got mine, but you’ll all get yours").[51]

In the pre-chorus, Swift's character asserts that she became wiser and hardened, "I rose up from the dead, I do it all the time."[51][52] The chorus is made up of the title, "Look what you made me do", repeated eight times;[17] Swift said that this part came when "the beat hit" and the production team decided to "edit out the rest of the words".[50] In the bridge, Swift's character threatens to give her enemies nightmares ("I don't trust nobody and nobody trusts me/ I’ll be the actress, starring in your bad dreams") and tells that the old version of her was dead, delivered through a sound effect that resembles a telephone call: "The old Taylor can't come to the phone right now/ Why?/ 'Cause she's dead";[46][53] Swift said that these lyrics were the most important part of the song.[50]

According to Swift, the song used some tropes from the TV series Game of Thrones: the "list of names" was inspired by Arya Stark's "kill list", and the "vibes" were inspired by the characters Cersei Lannister and Daenerys Targaryen.[54] Although "Look What You Made Me Do" does not mention any particular person, many publications interpreted some of its parts as references to the West–Kardashian controversy:[55] "I don't like your tilted stage",[17] "the old Taylor can't come to the phone right now".[46] Swift confirmed in a 2019 Rolling Stone cover story that the "phone call" lyric referred to "a stupid phone call [she] shouldn't have picked up".[56] There were also comments that some parts referenced Swift's fallout with Katy Perry[37][53] or her ex-boyfriends Calvin Harris and Tom Hiddleston.[57]

According to Carson, the songwriting contains some of the "storybook lyricism and fairytale tropes" that Swift had employed before, such as the imagery of kingdoms and dreams and the idea of revenge, which recalled songs like "Should've Said No" (2006), "Better than Revenge" (2010), and "Bad Blood" (2014);[37] Billboard's Tatiana Cirisano said that the dreams imagery evoked the songs from 1989 such as "Blank Space" or "Wildest Dreams".[53]
Critical reception

Upon release, "Look What You Made Me Do" polarized music critics.[33][58][59] Some considered the single a fierce return and an interesting move for Swift to reclaim her public narrative, whereas others found the production and themes vindictive, harsh, and off-putting.[60] USA Today said that the polarized reaction to the song illustrated Swift's position as a "ubiquitous cultural force".[61]

The Telegraph's Sarah Carson praised the song, deeming Swift and Antonoff's work as "blowing past the production clichés of clap tracks and hiccuped syllabic hooks that have proliferated across Top 40 fare in recent years with boldly inventive textures and fresh melodic, rhythmic and sonic accents". She also added how the track musically and sonically shifted alongside the lyrics.[37] Randy Lewis of the Los Angeles Times wrote a positive review of the song, saying: "The reverberating crescendo builds and ever more delicious is the wickedness of Swift's menacing protagonist", praising Swift for her successful embrace of the villain character the media has portrayed her as previous to the song's release.[38] Variety's Chris Willman also praised Swift's embrace of darker-styled pop music and the stylistic conflict between the song's pre-chorus and chorus.[52] Mark Harris, writing for Vulture, thought of Swift's song as a pop art anthem for the Trump era in how she reappropriates her public feuds as empowering badges of honor without acknowledging her responsibility or blame.[62]

Maura Johnston of The Guardian wrote a negative review of the song, faulting the "sloppy" lyrics and blaming Swift for not giving a clear context in the lyrics.[26] Lindsay Zoladz of The Ringer said, "Unleashed on a deeply confused public late Thursday night, the song is a strange collage of retro reference points: mid-aughts Gossip Girl placement pop, the soundtrack to Disney's live-action Maleficent, and — yes, really — Right Said Fred's "I'm Too Sexy", except devoid of the self-effacing humor and wit. Yes, the new Taylor Swift song just made me compliment Right Said Fred." Brittany Spanos of Rolling Stone believed that the song marked a continuation of the feud between Swift and rapper Kanye West; the latter had previously name-dropped Swift in his song "Famous" by using the line, "I feel like me and Taylor might still have sex / Why? / I made that bitch famous". The single was noted as being darker and angrier than what Swift had done before.[46][63] Meaghan Garvey from Pitchfork referred to it in a review as "a hardcore self-own" track.[64]

Retrospective reviews considered "Look What You Made Me Do" a career-defining song for Swift. In 2019, Slant listed "Look What You Made Me Do" as one of the 100 singles that defined the 2010s decade.[65] In 2023, Zoya Raza-Sheikh of The Independent opined that the single portrayed Swift as a "beleaguered" pop star, and functioned as a "clap back at the critics, media, and celebrity rivals who had celebrated her public "downfall" in 2016".[66]
Commercial performance

"Look What You Made Me Do" broke several streaming records upon release.[67][68] In the United States, "Look What You Made Me Do" debuted at number 77 on the Billboard Hot 100, with that week's chart capturing its first three days of airplay.[69] It also sold slightly under 200,000 digital copies within its first day of sales in the country, where it became the fastest-selling download since Ed Sheeran's "Shape of You".[70] One week later, the song ascended from number 77 to number one on the Hot 100 after its first full week of tracking, becoming the fifth-largest rise to the top position and Swift's fifth number-one single in the United States. Ending the record-tying 16-week reign of Luis Fonsi's "Despacito", "Look What You Made Me Do" became one of the most dominant number-one hits of all time, leading ahead of "Despacito" with more than double the Hot 100 points.[71]

The song also topped the nation's Streaming Songs chart with 84.4 million streams, becoming its most streamed song within a week by a female artist at the time and second overall behind the 103 million that Baauer's "Harlem Shake" gained in 2013. The track also had more weekly streams in the US than any other song in 2017. The song stayed atop the charts for three consecutive weeks, tying with American rapper Cardi B's "Bodak Yellow" as the longest-running female at the number one spot on the charts in 2017.[71]

With 353,000 copies sold in its first week, "Look What You Made Me Do" opened atop the US Digital Songs chart and had the country's biggest sales opening since Justin Timberlake's "Can't Stop the Feeling!" in 2016 as well as the best weekly sales for a song by a female artist since Adele's "Hello" in 2015. The track also became the country's first number-one song by a female artist since Sia's "Cheap Thrills" (both in 2016). It additionally was the first solo song by a female to top the US charts since "Hello".[71] It remained atop the Hot 100 and Streaming Songs charts for a second week with 114,000 copies sold and 61.2 million streams. That week, it was bumped to number two on the Digital Songs chart by Swift's track "...Ready for It?", which debuted at number one with 135,000 digital copies sold. As a result, Swift became the first artist to have two tracks sell over 100,000 digital copies in the nation within a week since Sheeran did so with "Shape of You" and "Castle on the Hill". It also became the first time a female had two songs within the top five of the Hot 100 since 2015 when Swift's songs "Blank Space" and "Shake It Off" respectively were at numbers four and five on the chart.[72] The single also topped the Mainstream Top 40 chart, becoming Swift's eighth single to do so.[73]

In the United Kingdom, "Look What You Made Me Do" sold 20,000 copies and was streamed 2.4 million times in less than a week.[74] The song debuted at the top of the UK Singles Chart on September 1, 2017 – for the week ending date September 7, 2017, with opening sales of 30,000 copies and 5.3 million streams within the week and becoming Swift's first chart-topping song in Britain.[75] It spent two weeks at the top spot.[76] As of December 2022, "Look What You Made Me Do" has sold over one million combined units in the UK.[77]

"Look What You Made Me Do" also debuted at number one on the Irish Singles Chart on September 1, 2017, and became Swift's first song to top the chart in Ireland. "Look What You Made Me Do" opened at number one in Australia on September 2, 2017, becoming her fifth track to top the ARIA Charts.[78] It spent another week at the summit.[79] The song has been certified seven-times platinum by the Australian Recording Industry Association (ARIA).[80] After debuting at number one on the Canadian Hot 100,[81] "Look What You Made Me Do" was also certified Platinum by Music Canada for shipments of 80,000 units on September 14, 2017.[82] In New Zealand, "Look What You Made Me Do" entered the number one spot on September 4, 2017, becoming Swift's fourth chart-topping single there.[83] In the Philippines, "Look What You Made Me Do" debuted at number seven on the Philippine Hot 100 in its first week. A week later, it ascended to the number one spot, ending the ten-week reign of "Despacito".[84]
Music video
Production and release

Preparation for the music video began in January 2017, while the shooting took place in May.[85][86] The dance was choreographed by Tyce Diorio, who had previously worked with Swift on the video for her 2014 single "Shake It Off".[85] Swift's makeup as a zombie was done by Bill Corso.[86] Post-production of the video lasted until the morning of its release date.[86] A 20-second music video teaser was released on Good Morning America on August 25.[87]

The song's music video premiered on August 27, 2017, at the 2017 MTV Video Music Awards.[88] The video broke the record for the most-watched video within 24 hours by achieving 43.2 million views on YouTube on its first day.[89] It topped the 27.7 million Vevo views Adele's "Hello" attracted in that timeframe, as well as the 36 million YouTube views of Psy's "Gentleman" video.[90][91][92] It was viewed at an average 30,000 times per minute in its first 24 hours, with views reaching over three million views per hour.[89] The video was named the fifth-best music video of 2017 by Rolling Stone[93] and the seventh-best music video of 2017 by Billboard.[94] In 2020, Parade ranked the video 20th on the list of 71 Best Music Videos of All Time.[95]
Synopsis
The bathtub scene in the music video. The diamonds used were said to be authentic and worth over $10 million, and a lone dollar bill can be seen.

Swift has said that part of the premise of the video is rooted in the idea that, "If everything you write about me was true, this is how ridiculous it would look."[96] It is a satirical send-up of media theories about her true intentions that have little validity. The video begins with an overhead shot of a cemetery before the camera zooms in on a grave with a headstone that reads "Here lies Taylor Swift's reputation." After that, a zombie Swift, wearing the dress from her "Out of the Woods" music video, crawls out of the grave before proceeding to dig another grave for her Met Gala 2014 self. The next scene shows Swift in a bathtub filled with diamonds, with a necklace spelling out "No" next to a ring, supposedly sending up tabloid press rumors of her past romantic relationships. She is then seen seated on a throne while snakes surround her and serve tea. Swift later crashes her golden Bugatti Veyron on a post and sings the song's chorus holding a Grammy as the paparazzi take photos. She is also seen swinging inside a golden cage, robbing a streaming company in a cat mask, and leading a motorcycle gang. Afterward, she gathers a group of women at "Squad U" and dances with a group of men in another room. Then, she is seen standing on top of the wing of a plane in an airport hangar, sawing off the wing in half and spray-painting "reputation" in pink on the side of the plane. At the video's climax, Swift is seen standing on a T-shaped throne mountain while clones of herself (from her past music videos, stage performances, and red-carpet appearances) struggle and fight against each other trying to reach her. The Swift at the top of the mountain stretches out her arms, and all the other Swifts fall off the mountain, while Swift from another scene picks up a phone and says "I'm sorry, the old Taylor can't come to the phone right now. Why? Oh, 'cause she's dead!" The video concludes with a scene of a line-up of surviving Swift clones bowing in the hangar while Swift stands and watches on the wing of the plane. The clones bicker with one another, describing each other as "so fake" and "playing the victim". The 2009 VMA Swift clone then says "I would very much like to be excluded from this narrative", resulting in the other Swifts yelling at her to "shut up!" in unison.[97][98]

Several scenes from the music video were compared to the works of Croatian singer Severina, particularly the scene with the group of women at "Squad U" and the scene with the T-shaped throne. The former was compared to her 2016 music video for "Silikoni", and the latter was compared to the performances from her 2013 Dobrodošao u Klub Tour.[99][100][101]
Analysis

The video contains numerous hidden meanings and references. In the opening scene, there is a subtle "Nils Sjöberg" tombstone shown when Swift is digging up a grave, referencing the pseudonym she used for a songwriting credit on Calvin Harris' 2016 single "This Is What You Came For".[102] Similarly, Swift—masked as a cadaveric version of herself in the "Out of the Woods" music video—was shown digging a grave for herself in the gown worn to the 2014 Met Gala.[103] The zombie Swift rising from her supposed "grave" is also speculated to be a subtle reference to Michael Jackson's Thriller music video, which showcases a zombie rising from their grave very similarly to the position Swift was in.[104][105] A single dollar bill in the bathtub full of diamonds that she bathes in was also speculated to symbolize the dollar she was awarded for winning a sexual assault trial earlier in 2017.[102] Interpretations for the bathtub scene were contrasting. Some believe that it is a response to media statements teasing that she "cries in a marble bathtub surrounded by pearls" with the necklace spelling "no" next to a ring sending up tabloid media rumors of her relationships.[106] Others speculate that the bathtub scene is a jibe at Kim Kardashian, the then-wife of Swift's long-time feuding partner, Kanye West. Some viewers took the scene as a reference to Kardashian's 2016 robbery, in which she was robbed of jewelry worth over $10 million while held at gunpoint at a hotel in Paris, France.[107][108]

In a separate scene, Swift is shown sitting atop a golden throne, where a carving of the phrase "Et tu, Brute?" could be seen on the armrest, a reference to William Shakespeare's drama Julius Caesar.[102] Swift's infamous title as a "snake" during her hiatus[109] was also represented when a snake slithers onto the throne to serve Swift some tea. The scene where Swift's car crashes and is surrounded by paparazzi was speculated by some to be a jab at Katy Perry, as Swift's hairstyle is similar to Perry's in the scene and the car crash itself is reminiscent of the one in Perry's music video for "Unconditionally" (2013). The sports car was also suspected to be a reminder of a car in Perry's "Waking Up in Vegas" (2009) video, which Kahn also directed. However, given the video's theme of mocking the media, the car crash scene likely makes fun of the theory that Swift's real fallout with Perry was a dramatized act for publicity and album material. Swift is ridiculing the idea that she would damage her friendships for business gains, with the car crash being a metaphor for her feud with Perry and the Grammy Award in her hands in the wreckage symbolizing the awards won from the songs "inspired" by the aforementioned feud. Swift's withdrawal of her entire music catalog from streaming services and the media's claims that she was doing this for greed and to start her streaming company was hinted at when Swift and her crew robbed a streaming company's money vault in the video.[110]

Swift leading an army of tall, skinny, and robotic women at a "Squad U" gathering poked fun at the media's accusations that her close group of friends were artificial and had unrealistic body shapes.[102] During the second chorus, Swift is seen with eight men, each of whom revealed an "I Heart TS" crop top after unbuttoning a jacket at her command. This scene mocks the idea that Swift forced her then-boyfriend Tom Hiddleston to wear an "I Heart TS" tank top. During the bridge, Swift stands on a mountain of clones of her past selves, which reiterates that she is leaving behind her "America's Sweetheart" image and embracing her newfound role as an evil "snake". The clones are wearing various noteworthy outfits that Swift herself previously wore. The shirt that her "You Belong with Me" music video clone wears, however, is slightly different from the original one: this time, the names of her current close friends are scribbled on it.[110]

The video's ending features an assembly of "old Taylors" in front of a private jet who are talking amongst one another and making snide references to the many false and exaggerated media portrayals of her throughout her career. These include claims that Swift fakes her classic surprised face at award shows; that her "nice girl" façade masks a truly mean, manipulative personality; accusations that Swift always plays the victim instead of taking responsibility for her actions and decisions; and numerous mentions of her 2016 feud with Kanye West and Kim Kardashian, ignited by the release of his 2016 song "Famous". Examples include the "that bitch" line in "Famous" which Swift had disapproved of, and Kardashian illegally recording and editing Swift's phone call with West.[111] In June 2016, discussing the relationship between her and West after the release of "Famous", Swift wrote on Instagram, "I would very much like to be excluded from this narrative."[112] The same line is spoken by the 2009 MTV Video Music Awards Taylor clone just before the video ends.[113][114] She is wearing the same outfit Swift had worn during the actual 2009 MTV Video Music Awards when West infamously interrupted her acceptance speech for the Best Female Video award and ignited tensions between the two for the first time.[115]
Live performances and other uses
Taylor Swift singing on a microphone, dressed in a black suit with golden accents
Swift singing on a microphone dressed in an asymmetrical bodysuit with snake embroidery
Swift performing "Look What You Made Me Do" on her Reputation Stadium Tour in 2018 (left) and The Eras Tour in 2023 (right)

Swift performed "Look What You Made Me Do" live for the first time as part of the KIIS-FM Jingle Ball 2017 on December 1, 2017, in Inglewood, California.[116] Two days later, Swift returned onstage to perform the song again as part of 99.7 Now!'s Poptopia in San Jose, California with the same setlist.[117] The next week Swift sang the song on three other occasions; the B96 Chicago and Pepsi Jingle Bash 2017 in Chicago, the Z100 Jingle Ball 2017 in New York City, and the Jingle Bell Ball 2017 in London.[118][119][120]

The song was also a regular part of her set list for the Reputation Stadium Tour, with a tilted throne and golden snakes; while there are snakes on the high screen in the back during the part where she sings, "I don't trust nobody and nobody trust me, I'll be the actress starring in your bad dreams", a large floating cobra appears onstage with the line from the bridge announcing the death of the "Old Taylor" spoken by comedian Tiffany Haddish.[121][122] Swift included "Look What You Made Me Do" on the set list of the Eras Tour (2023–2024).[123] On the Eras Tour, while Swift is performing "Look What You Made Me Do," she is surrounded by her backup singers and dancers, who are dressed as various past versions of herself and trapped inside large clear boxes. [124] This iconography is similar to that which was used in the music video. Prior to the announcement and release of Speak Now (Taylor's Version), many fans noticed Taylor pounding on the box with the dancer dressed in the Speak-Now-era purple halter dress, and speculated that this interaction signaled that Speak Now would be the next album to be rerecorded. [125]

ABC used "Look What You Made Me Do" in a promotional video for its Shonda Rhimes' Thursday line-up an hour after its release.[126] ESPN used the song in Saturday Night Football advertisements for the season-opening game between Alabama and Florida State, which was aired on ABC on September 2 along with her other song "...Ready for It?".[127] In the South Park episode "Moss Piglets", the water bears in Timmy and Jimmy's experiment for the science fair dance to the song in response to Swift's singing. The song was used in the trailer for the 2019 comedy film Murder Mystery.[128] American actress Reese Witherspoon performs the song for the jukebox musical film, Sing 2, in the role of Rosita.[129]
Jack Leopards & the Dolphin Club cover

A cover version of "Look What You Made Me Do" was recorded by the band Jack Leopards & the Dolphin Club, and produced by Antonoff and Nils Sjöberg, the latter being a pseudonym that Swift first used as a co-writer for the song "This Is What You Came For" by Calvin Harris featuring Rihanna. The cover was featured in the opening credits of "Beautiful Monster", an episode of the television show Killing Eve that aired on May 24, 2020, and subsequently released on digital music platforms. There is no documentation of the band's existence before the release of the cover,[130] and it was speculated the person singing was Swift's brother Austin Swift.[131][132] Fans also interpreted the cover to be Swift's way of bypassing potential licensing issues with her former label Big Machine Records and its owner Scooter Braun, with whom Swift is involved in a dispute regarding Braun's acquisition of the label and, subsequently, the master recordings of her back catalogue.[133]
"Taylor's Version" re-recording

On August 23, 2023, "Look What You Made Me Do (Taylor's Version)," from the unreleased Reputation (Taylor's Version) was teased in a trailer for the Amazon Prime Video series Wilderness, released on September 15, 2023.[134]
Accolades
Year 	Organization 	Award 	Result 	Ref.
2017 	MTV Europe Music Awards 	Best Music Video 	Nominated 	[135]
NRJ Music Awards 	Video of the Year 	Nominated 	[136]
2018 	MTV Millennial Awards Brazil 	Best International Hit 	Nominated 	[137]
NME Awards 	Best Music Video 	Nominated 	[138]
iHeartRadio Music Awards 	Best Music Video 	Nominated 	[139]
Best Lyrics 	Nominated
Nickelodeon Kids' Choice Awards 	Favorite Song 	Nominated 	[140]
Myx Music Award 	Favourite International Video 	Nominated 	[141]
Radio Disney Music Awards 	Song of the Year 	Nominated 	[142]
Best Song To Lip-Sync To 	Nominated
Teen Choice Awards 	Choice Song by a Female Artist 	Nominated 	[143]
Hito Music Awards 	Best Western Song 	Won 	[144]
MTV Video Music Awards 	Best Art Direction 	Nominated 	[145]
Best Editing 	Nominated
Best Visual Effects 	Nominated
BMI London Awards 	Pop Award 	Won 	[146]
Guinness World Records 	Most Streamed Track in One Week (Female) 	Won 	[147]
Most Watched Video Online in 24 Hours 	Won
Most Streamed Track on Spotify in the First 24 Hours 	Won
Most Watched Video on VEVO in 24 Hours 	Won
2019 	BMI Awards 	Award Winning Song 	Won 	[148]
Publisher of the Year 	Won
TEC Awards 	Best Record Production / Single or Track 	Nominated 	[149]
Credits and personnel

Credits are adapted from the liner notes of Reputation.[35]

    Taylor Swift – vocals, songwriter, producer
    Jack Antonoff – producer, songwriter, programming, instruments
    Richard Fairbrass – songwriter, interpolation
    Fred Fairbrass – songwriter, interpolation
    Bob Manzoli – songwriter, interpolation
    Laura Sisk – engineer
    Serban Ghenea – mixing
    John Hanes – mix engineer
    Randy Merrill – mastering
    Evan Smith – saxophones
    Victoria Parker – violins
    Phillip A. Peterson – cellos

Charts
Weekly charts
Chart (2017–2018) 	Peak
position
Argentina Anglo (Monitor Latino)[150] 	4
Australia (ARIA)[151] 	1
Austria (Ö3 Austria Top 40)[152] 	2
Belgium (Ultratop 50 Flanders)[153] 	8
Belgium (Ultratop 50 Wallonia)[154] 	39
Bulgaria (PROPHON)[155] 	9
Canada (Canadian Hot 100)[156] 	1
Canada AC (Billboard)[157] 	33
Canada CHR/Top 40 (Billboard)[158] 	10
Canada Hot AC (Billboard)[159] 	10
CIS Airplay (TopHit)[160] 	69
Colombia (Promúsica)[161] 	18
Colombia (National-Report)[162] 	46
Costa Rica (Monitor Latino)[163] 	6
Croatia (HRT)[164] 	1
Czech Republic (Rádio – Top 100)[165] 	39
Czech Republic (Singles Digitál Top 100)[166] 	1
Denmark (Tracklisten)[167] 	12
Ecuador (National-Report)[168] 	27
El Salvador (Monitor Latino)[169] 	7
Euro Digital Song Sales (Billboard)[170] 	1
Finland (Suomen virallinen lista)[171] 	8
France (SNEP)[172] 	26
Germany (GfK)[173] 	3
Greece Digital Songs (Billboard)[174] 	1
Greece International (IFPI)[175] 	2
Honduras (Monitor Latino)[176] 	15
Hungary (Single Top 40)[177] 	3
Ireland (IRMA)[178] 	1
Israel (Media Forest)[179] 	1
Italy (FIMI)[180] 	10
Japan (Japan Hot 100)[181] 	7
Lebanon (Lebanese Top 20)[182] 	1
Luxembourg Digital Song Sales (Billboard)[183] 	5
Malaysia (RIM)[184] 	1
Mexico Airplay (Billboard)[185] 	10
Netherlands (Dutch Top 40)[186] 	7
Netherlands (Single Top 100)[187] 	13
New Zealand (Recorded Music NZ)[83] 	1
Norway (VG-lista)[188] 	6
Panama (Monitor Latino)[189] 	12
Philippines (Philippine Hot 100)[84] 	1
Portugal (AFP)[190] 	4
Russia Airplay (TopHit)[191] 	65
Romania (Airplay 100)[192] 	64
Scotland (OCC)[193] 	1
Slovakia (Singles Digitál Top 100)[194] 	1
Slovenia (SloTop50)[195] 	26
South Korea (Circle)[196] 	92
Spain (PROMUSICAE)[197] 	12
Sweden (Sverigetopplistan)[198] 	7
Switzerland (Schweizer Hitparade)[199] 	6
UK Singles (OCC)[200] 	1
Uruguay (Monitor Latino)[201] 	14
US Billboard Hot 100[202] 	1
US Adult Contemporary (Billboard)[203] 	19
US Adult Pop Airplay (Billboard)[204] 	7
US Dance Club Songs (Billboard)[205] 	9
US Dance/Mix Show Airplay (Billboard)[206] 	3
US Pop Airplay (Billboard)[207] 	1
US Rhythmic (Billboard)[208] 	20
Venezuela Anglo (Monitor Latino)[209] 	5
Venezuela (National-Report)[210] 	66
Chart (2024) 	Peak
position
Singapore (RIAS)[211] 	26
	
Year-end charts
Chart (2017) 	Position
Australia (ARIA)[212] 	61
Austria (Ö3 Austria Top 40)[213] 	70
Brazil (Pro-Música Brasil)[214] 	129
Canada (Canadian Hot 100)[215] 	29
Costa Rica (Monitor Latino)[216] 	35
El Salvador (Monitor Latino)[217] 	12
Germany (Official German Charts)[218] 	81
Honduras (Monitor Latino)[219] 	28
Hungary (Single Top 40)[220] 	48
Hungary (Stream Top 40)[221] 	51
Netherlands (Dutch Top 40)[222] 	87
Nicaragua (Monitor Latino)[223] 	80
Paraguay (Monitor Latino)[224] 	92
Portugal (AFP)[225] 	89
UK Singles (Official Charts Company)[226] 	89
US Billboard Hot 100[227] 	39
US Adult Top 40 (Billboard)[228] 	42
US Mainstream Top 40 (Billboard)[229] 	41
Chart (2018) 	Position
South Korea International (Gaon)[230] 	87

Certifications
Certifications for "Look What You Made Me Do" Region 	Certification 	Certified units/sales
Australia (ARIA)[80] 	7× Platinum 	490,000‡
Austria (IFPI Austria)[231] 	Platinum 	30,000‡
Belgium (BEA)[232] 	Gold 	10,000‡
Brazil (Pro-Música Brasil)[233] 	2× Diamond 	320,000‡
Canada (Music Canada)[82] 	3× Platinum 	240,000‡
Denmark (IFPI Danmark)[234] 	Gold 	45,000‡
France (SNEP)[235] 	Gold 	66,666‡
Germany (BVMI)[236] 	Gold 	200,000‡
Italy (FIMI)[237] 	Platinum 	50,000‡
New Zealand (RMNZ)[238] 	3× Platinum 	90,000‡
Norway (IFPI Norway)[239] 	Platinum 	60,000‡
Poland (ZPAV)[240] 	2× Platinum 	40,000‡
Portugal (AFP)[241] 	Platinum 	10,000‡
Spain (PROMUSICAE)[242] 	Platinum 	60,000‡
Sweden (GLF)[243] 	2× Platinum 	80,000‡
United Kingdom (BPI)[244] 	2× Platinum 	1,200,000‡
United States (RIAA)[245] 	4× Platinum 	4,000,000‡

‡ Sales+streaming figures based on certification alone.
Release history
Release dates and formats for "Look What You Made Me Do" Region 	Date 	Format 	Label 	Ref.
Various 	August 24, 2017 	Streaming 	Big Machine 	[21]
August 25, 2017 	Digital download 	[23]
Italy 	Radio airplay 	Universal 	[246]
United States 	August 29, 2017 	Contemporary hit radio 	Big Machine 	[28]
Germany 	October 27, 2017 	CD single 	Universal 	[29]
See also

    List of most-streamed songs on Spotify
    List of Billboard Hot 100 number-one singles of 2017
    List of number-one Billboard Streaming Songs of 2017
    List of Billboard Mainstream Top 40 number-one songs of 2017
    List of UK Singles Chart number ones of the 2010s
    List of Canadian Hot 100 number-one singles of 2017
    List of number-one singles of 2017 (Australia)
    List of number-one digital tracks of 2017 (Australia)
    List of number-one streaming tracks of 2017 (Australia)
    List of number-one singles of 2017 (Ireland)
    List of number-one songs of 2017 (Malaysia)
    List of number-one singles from the 2010s (New Zealand)

References

McNutt 2020, p. 78–79.
"Taylor Swift Returns to Spotify On the Day Katy Perry's Album Comes Out". BBC News. June 9, 2017. Archived from the original on June 9, 2017.
Unterberger, Andrew (July 6, 2018). "While You Weren't Looking, Taylor Swift Scored Her Biggest Reputation Radio Hit". Billboard. Archived from the original on July 7, 2018. Retrieved October 11, 2020.
Lynskey, Dorian (December 13, 2017). "Taylor Swift's Complex Reputation". British GQ. Archived from the original on January 25, 2023. Retrieved January 25, 2023.
Ryan, Patrick (November 9, 2017). "5 Things Taylor Swift's Past USA Today Interviews Tell Us About Her Reputation Era". USA Today. Archived from the original on November 10, 2017. Retrieved November 9, 2017.
Voght, Kara (December 23, 2022). "The Year Everyone Realized They Were Wrong About Taylor Swift vs. Kanye West". Rolling Stone. Retrieved January 1, 2023.
Lipshutz, Jason (August 24, 2017). "Taylor Swift's Reputation, Intact: Expect the Script to Once Again Get Flipped". Billboard. Retrieved January 3, 2023.
Hiatt, Brian (September 18, 2019). "Taylor Swift: The Rolling Stone Interview". Rolling Stone. Retrieved January 7, 2023.
Aguirre, Abby (August 8, 2019). "Taylor Swift on Sexism, Scrutiny, and Standing Up for Herself". Vogue. Archived from the original on August 10, 2019. Retrieved August 16, 2019.
Vincent, Alice (November 3, 2017). "Taylor Swift: The Rise, Fall and Re-Invention of America's Sweetheart". The Daily Telegraph. Archived from the original on January 10, 2022. Retrieved January 1, 2023.
Jones, Nate (July 21, 2016). "When Did the Media Turn Against Taylor Swift?". Vulture. Archived from the original on November 18, 2021. Retrieved January 3, 2023.
Yahr, Emily (November 15, 2017). "Taylor Swift Avoided – and Mocked – the Media with Reputation. And It Worked". The Washington Post. Retrieved November 15, 2017.
Garcia, Patricia (August 25, 2017). "With 'Look What You Made Me Do', Taylor Swift Inserts Herself Back Into the Narrative". Vogue. Retrieved November 24, 2023.
Wilkinson 2017, p. 444.
Kaufman, Gil (August 21, 2017). "Taylor Swift Ends Social Media Blackout With Cryptic Reptile Tail Tease". Billboard. Retrieved January 9, 2023.
Bengtsson & Edlom 2023.
Coscarelli, Joe; Pareles, Jon; Caramanica, Jon; Morris, Wesley; Ganz, Caryn (August 25, 2017). "Taylor Swift Goes to a Darker Place: Discuss". The New York Times. Archived from the original on January 11, 2023. Retrieved January 11, 2023.
Cirisano, Tatiana (August 23, 2017). "Taylor Swift's Third Reptile-Themed Teaser Finally Reveals Snake's Face". Billboard. Retrieved January 11, 2023.
Gaca, Anna (May 9, 2018). "Taylor Swift Talks 'Snakes' and Kim Kardashian at Reputation Tour Opener". Spin. Archived from the original on January 24, 2023. Retrieved January 24, 2023.
Stolworthy, Jacob (August 23, 2017). "Taylor Swift Reputation: Singer Unveils New Album Name, Cover and Release Date". The Independent. Archived from the original on August 25, 2017. Retrieved August 23, 2017.
Aswad, Jem (August 24, 2017). "Taylor Swift's New Single, 'Look What You Made Me Do,' Arrives (Listen)". Variety. Archived from the original on August 28, 2017. Retrieved August 29, 2017.
Bryant, Kenzie (August 28, 2017). "Taylor Swift Is Ready to Come Out of Hiding, But Are We Ready for Her?". Vanity Fair. Retrieved November 24, 2023.
"Look What You Made Me Do – Single". iTunes Store. Archived from the original on August 25, 2017. Retrieved August 25, 2017.
Whittaker, Alexandra (August 25, 2017). "Here's Every Lyric from Taylor Swift's New Song 'Look What You Made Me Do'". InStyle. Archived from the original on August 25, 2017. Retrieved August 25, 2017.
Cauterucci, Christina (August 25, 2017). "Why Taylor Swift's 'Look What You Made Me Do' Is Such a Colossal Bummer". Slate. Retrieved November 24, 2023.
Johnston, Maura (August 24, 2017). "Acid gossip that borrows from better songs – Taylor Swift: 'Look What You Made Me Do' review". The Guardian. Archived from the original on September 12, 2017. Retrieved August 25, 2017.
Knapp, JD (August 26, 2017). "Taylor Swift's 'Look What You Made Me Do' Lyric Video Breaks 24-Hour Record". Variety. Archived from the original on August 27, 2017. Retrieved August 26, 2017.
"Top 40/M Future Releases". All Access. All Access Music Group. Archived from the original on August 29, 2017. Retrieved August 29, 2017.
Swift, Taylor (October 27, 2017). "Look What You Made Me Do (2-Track)". Amazon Germany (in German). Archived from the original on February 10, 2023. Retrieved October 14, 2017.
"Everything we know about Taylor Swift's comeback". BBC. Archived from the original on March 27, 2020. Retrieved July 3, 2020.
Nevins, Jake (August 23, 2017). "Taylor Swift announces new album, Reputation, for November release". The Guardian. Archived from the original on November 7, 2017. Retrieved July 3, 2020.
Carvan, Tabitha (May 17, 2023). "Why Reputation era Taylor Swift was my favourite Taylor Swift". The Sydney Morning Herald. Retrieved September 10, 2024.
Ryan, Patrick (August 25, 2017). "Look What You Made Us Do: Critics Slam Taylor Swift's Comeback Single". USA Today. Archived from the original on August 25, 2017. Retrieved August 26, 2017.
Battan, Carrie (August 25, 2017). "Taylor Swift's New Song, 'Look What You Made Me Do'". The New Yorker. Archived from the original on September 3, 2017. Retrieved September 30, 2019.
Reputation (CD booklet). Taylor Swift. Big Machine Records. 2017.
"Taylor Swift – 'Look What You Made Me Do' Sheet Music (Digital Download)". Musicnotes.com. Sony/ATV Music Publishing. August 25, 2017. Archived from the original on August 27, 2017. Retrieved August 27, 2017.
Carson, Sarah (August 25, 2017). "Taylor Swift, 'Look What You Made Me Do', review: 'Swift has painted herself as a villain, and triumphed'". The Daily Telegraph. Retrieved January 11, 2024.
Lewis, Randy (August 24, 2017). "Taylor Swift drops 'Look What You Made Me Do' and it's aggressive". Los Angeles Times. Archived from the original on August 25, 2017. Retrieved August 25, 2017.
Jenkins, Craig (November 10, 2017). "Taylor Swift's Reputation Fixates on Big Enemies and Budding Romance". Vulture. Retrieved December 6, 2023.
Lockett, Dee (August 25, 2017). "Taylor Swift Declares the Old Taylor Dead on New Song, 'Look What You Made Me Do'". Vulture. Archived from the original on August 25, 2017. Retrieved August 25, 2017.
"Taylor Swift releases new song 'Look What You Made Me Do'". Daily News. New York. August 24, 2017. Archived from the original on August 25, 2017. Retrieved August 24, 2017.
Grow, Cory (August 25, 2017). "Right Said Fred on Taylor Swift's 'Cynical' 'Look What You Made Me Do'". Rolling Stone. Archived from the original on August 25, 2017. Retrieved August 26, 2017.
Sheffield, Rob (October 6, 2017). "Taylor Swift's New Album Reputation: Everything We Know, Everything We Want". Rolling Stone. Archived from the original on October 20, 2017. Retrieved October 20, 2017.
Kinbbs, Kate (August 21, 2019). "Ten Years of Taylor Swift: How the Pop Star Went From Sweetheart to Snake (and Back Again?)". The Ringer. Archived from the original on May 25, 2022. Retrieved June 8, 2022.
Gotrich, Lars; Lorusso, Marissa; McKenna, Lyndsey (August 25, 2017). "Taylor Swift Can't Be The Victim And The Villain". NPR. Archived from the original on October 22, 2022. Retrieved October 22, 2022.
Spanos, Brittany (August 25, 2017). "Taylor Swift Releases Apparent Kanye West Diss Song 'Look What You Made Me Do'". Rolling Stone. Archived from the original on August 25, 2017. Retrieved August 25, 2017.
"Singles Club Christmas special: The biggest tracks of 2017 revisited". Fact. December 19, 2017. Retrieved November 24, 2023.
Zaleski 2024, p. 139.
"The Biggest Influences on Pop in the 2010s". Rolling Stone. December 23, 2019. Retrieved January 11, 2024.
Mastrogiannis, Nicole (November 11, 2017). "Taylor Swift's iHeartRadio reputation Release Party: Everything We Learned". WKST-FM. Retrieved September 12, 2024.
Guan, Frank (August 25, 2017). "Review: Taylor Swift's New Single 'Look What You Made Me Do' Is Dead on Arrival". Vulture. Retrieved September 12, 2024.
Willman, Chris (August 25, 2017). "Song Review: Taylor Swift's 'Look What You Made Me Do'". Variety. Archived from the original on August 25, 2017. Retrieved August 25, 2017.
Cirisano, Tatiana (August 25, 2017). "Decoding 7 Revealing Lyrics on Taylor Swift's 'Look What You Made Me Do'". Billboard. Retrieved September 12, 2024.
Suskind, Alex (May 9, 2019). "Taylor Swift reveals how Game of Thrones (and Arya's kill list) inspired reputation". Entertainment Weekly. Archived from the original on May 9, 2019. Retrieved May 9, 2019.
Spanos, Brittany (August 25, 2017). "Are Fans Ready for Taylor Swift's Darkest Era Yet?". Rolling Stone. Retrieved September 12, 2024.
Hiatt, Brian (September 18, 2019). "Taylor Swift: The Rolling Stone Interview". Rolling Stone. Retrieved September 12, 2024.
Stubbs, Dan (August 25, 2017). "Taylor Swift reclaims the snake on new single 'Look What You Made Me Do'". NME. Retrieved September 12, 2024.
Holterman, Alexandra (August 25, 2017). "Taylor Swift's Lead Singles From Each Album: Which Do You Think Is Best? Vote!". Billboard. Archived from the original on June 21, 2018. Retrieved July 8, 2018.
Holterman, Alexandra (August 30, 2017). "'Look What You Made Me Do' Director Defends Taylor Swift, Accuses Public of 'Double Standards'". Billboard. Archived from the original on August 31, 2017. Retrieved August 31, 2017.
Spanos, Brittany (December 7, 2018). "Taylor Swift's Reputation Is Grammys' Biggest Snub". Rolling Stone. Retrieved November 24, 2023.
McDermott, Maeve (August 25, 2017). "Taylor Swift Goes Full Psycho Pop in New Song 'Look What You Made Me Do'". USA Today. Archived from the original on August 25, 2017. Retrieved August 25, 2017.
Harris, Mark (August 30, 2017). "Taylor Swifts Look What You Made Me Do Is a Pure Piece of Trump-Era Pop Art". Vulture. Archived from the original on September 4, 2017. Retrieved September 5, 2017.
Weatherby, Taylor (August 25, 2017). "Taylor Swift Drops Dark New Song 'Look What You Made Me Do'". Billboard. Archived from the original on August 25, 2017. Retrieved August 25, 2017.
Garvey, Meaghan (August 25, 2017). "'Look What You Made Me Do' by Taylor Swift Review". Pitchfork. Archived from the original on August 25, 2017. Retrieved August 25, 2017.
"The 100 Best Singles of the 2010s". Slant. January 1, 2020. Archived from the original on January 1, 2020. Retrieved January 1, 2020.
Zoya, Raza-Sheikh (February 3, 2023). "How Taylor Swift mastered the singer-songwriter blueprint". The Independent. Archived from the original on June 1, 2023. Retrieved June 1, 2023.
"Taylor Swift breaks Spotify, YouTube records". CBS News. August 27, 2017. Retrieved November 24, 2023.
Flanagan, Andrew (September 5, 2017). "Taylor Swift's 'Look What You Made Me Do' Breaks Records, Stops 'Despacito' Short". NPR. Retrieved November 24, 2023.
Trust, Gary (August 28, 2017). "Taylor Swift's 'Look What You Made Me Do' Headed for No. 1 on Billboard Hot 100 Next Week". Billboard. Archived from the original on August 29, 2017. Retrieved August 28, 2017.
Caulfield, Keith (August 26, 2017). "Taylor Swift's 'Look' Heads for Half-Million Sales, Biggest Week Since Adele's 'Hello'". Billboard. Archived from the original on August 29, 2017. Retrieved August 29, 2017.
Trust, Gary (September 5, 2017). "Taylor Swift's 'Look What You Made Me Do' Leaps to No. 1 on Hot 100 With Top Streaming & Sales Week of 2017". Billboard. Archived from the original on September 5, 2017. Retrieved September 5, 2017.
Trust, Gary (September 11, 2017). "Taylor Swift at Nos. 1 & 4 on Billboard Hot 100, as Cardi B Moves Up to No. 2". Billboard. Archived from the original on September 12, 2017. Retrieved September 13, 2017.
Trust, Gary (October 16, 2017). "Taylor Swift's 'Look What You Made Me Do' Tops Pop Songs Chart". Billboard. Archived from the original on October 16, 2017. Retrieved October 17, 2017.
White, Jack (August 29, 2017). "Taylor Swift could earn her first UK Number 1 single this Friday". Official Charts Company. Archived from the original on August 29, 2017. Retrieved August 29, 2017.
White, Jack (September 1, 2017). "Taylor Swift scores first Number 1 on the Official Singles Chart with 'LWYMMD'". Official Charts Company. Archived from the original on June 3, 2013. Retrieved September 1, 2017.
White, Jack (September 15, 2017). "Sam Smith scoops his sixth UK Number 1 single with Too Good at Goodbyes". Official Charts Company. Archived from the original on November 18, 2017. Retrieved September 15, 2017.
Copsey, Rob (April 26, 2019). "Taylor Swift's Official Top 20 biggest singles in the UK revealed". Official Charts Company. Archived from the original on March 21, 2020. Retrieved April 26, 2019.
"Taylor Swift Scores Fifth No. 1 Single". Australian Recording Industry Association. September 2, 2017. Archived from the original on September 2, 2017. Retrieved September 2, 2017.
"Taylor Swift holds No. 1 for second week". Australian Recording Industry Association. Archived from the original on September 17, 2017. Retrieved September 16, 2017.
"ARIA Charts – Accreditations – 2024 Singles" (PDF). Australian Recording Industry Association. Retrieved June 14, 2024.
"Canadian Music: Top 100 Songs Chart". Billboard. September 16, 2017. Archived from the original on September 15, 2017. Retrieved September 16, 2017.
"Canadian single certifications – Taylor Swift – Look What You Made Me Do". Music Canada. Retrieved May 18, 2018.
"Taylor Swift – Look What You Made Me Do". Top 40 Singles. Retrieved September 1, 2017.
"BillboardPH Hot 100". Billboard Philippines. September 18, 2017. Archived from the original on September 18, 2017. Retrieved September 18, 2017.
Murphy, Desiree (August 30, 2017). "Exclusive: Todrick Hall Reveals Secrets From Taylor Swift's 'Look What You Made Me Do' Music Video". Entertainment Tonight. Archived from the original on August 30, 2017. Retrieved August 25, 2017.
Murphy, Desiree (August 30, 2017). "Taylor Swift's 'Look What You Made Me Do' Video: Everything We Know About the Snakes, Diamonds, Dancing & More". Entertainment Tonight. Archived from the original on August 31, 2017. Retrieved August 30, 2017.
Maureen Lee Lenker (August 25, 2017). "Snake slithers up Taylor Swift's throne in dark, glamorous music video". Entertainment Weekly. Archived from the original on August 31, 2017. Retrieved August 31, 2017.
Strauss, Matthew; Minsker, Evan (August 27, 2017). "Watch Taylor Swift's "Look What You Made Me Do" Video". Pitchfork. Archived from the original on August 28, 2017. Retrieved August 27, 2017.
Cirisano, Tatiana (August 29, 2017). "Taylor Swift Tops PSY's 24-Hour YouTube Record With 'Look What You Made Me Do'". Billboard. Archived from the original on August 30, 2017. Retrieved August 30, 2017.
"Taylor Swift's 'Look What You Made Me Do' Smashes YouTube's 24-Hour Record, Crushing Psy". Variety. August 29, 2017. Archived from the original on August 29, 2017. Retrieved August 29, 2017.
Lewis, Randy (August 29, 2017). "Taylor Swift's 'Look What You Made Me Do' video bashes another YouTube record". Los Angeles Times. Archived from the original on August 29, 2017. Retrieved August 28, 2017.
McIntyre, Hugh (August 31, 2017). "Taylor Swift's 'Look What You Made Me Do' Video Hit 100 Million Views in Less Than Four Days". Forbes. Archived from the original on September 1, 2017. Retrieved August 31, 2017.
Ducker, Eric (December 27, 2017). "10 Best Music Videos of 2017". Rolling Stone. Archived from the original on September 30, 2019. Retrieved September 30, 2019.
"The 10 Best Music Videos of 2017: Critics' Picks". Billboard. Archived from the original on September 30, 2019. Retrieved September 30, 2019.
Sager, Jessica (July 6, 2020). "Stop, Look and Listen—These Are the 71 Best Music Videos Of All Time". Parade. Archived from the original on February 25, 2021. Retrieved March 30, 2021.
Apple Music
Spanos, Brittany (August 27, 2017). "Watch Taylor Swift Mock Herself in Dark 'Look What You Made Me Do' Video". Rolling Stone. Archived from the original on August 30, 2017. Retrieved August 31, 2017.
Yahr, Emily (August 28, 2017). "Taylor Swift knows you've been making fun of her. Here's how her new video responds". The Washington Post. Archived from the original on September 4, 2017. Retrieved August 28, 2017.
B. M. (August 31, 2017). "Taylor Swift iskopirala Severinu? Spot koji je uzdrmao svijet inspirirala Seve?". Dnevnik.hr (in Croatian). Archived from the original on August 22, 2022. Retrieved August 22, 2022.
J. Mi (August 31, 2017). "VIDEO Taylor Swift iskopirala Severinu?". Index.hr (in Croatian). Archived from the original on August 22, 2022. Retrieved August 22, 2022.
"Obožavatelji imaju teoriju: Taylor Swift je u novom spotu iskopirala Severinu?". RTL.hr. September 1, 2017. Archived from the original on August 22, 2022. Retrieved August 22, 2022.
Whitehead, Mat (August 28, 2017). "8 Things You Might Have Missed in Taylor's 'Look What You Made Me Do' Video". HuffPost. Archived from the original on August 31, 2017. Retrieved August 31, 2017.
Acuna, Kirsten (August 30, 2017). "We decoded all the different looks in Taylor Swift's music video — here's what they mean". Insider. Archived from the original on February 10, 2023. Retrieved July 6, 2020.
"Here Are All of the References Taylor Swift Made in the "Look What You Made Me Do" Video". Vogue. August 28, 2017. Retrieved November 2, 2023.
"All the References to Better Music Videos in Taylor Swift's "Look What You Made Me Do"". W Magazine. August 28, 2017. Retrieved November 2, 2023.
Jones, Marcus (August 28, 2017). "Here's Why Some People Are Mad at Taylor Swift's New Video". BuzzFeed. Archived from the original on August 30, 2017. Retrieved August 31, 2017.
O'Connor, Roisin (August 30, 2017). "Taylor Swift's diamond bath in the LWYMMD video was worth more than $10 million". The Independent. Archived from the original on February 26, 2018. Retrieved February 25, 2018.
Serhan, Yasmeen (October 3, 2016). "The $10 Million Robbery of Kim Kardashian West". The Atlantic. Archived from the original on February 26, 2018. Retrieved February 25, 2018.
France, Lisa Respers (August 23, 2017). "Taylor Swift and snakes: The backstory". CNN. Archived from the original on August 30, 2017. Retrieved August 31, 2017.
Chen, Joyce (August 28, 2017). "Taylor Swift's 'Look What You Made Me Do' Video Decoded: 13 Things You Missed". Rolling Stone. Archived from the original on August 30, 2017. Retrieved August 31, 2017.
France, Lisa (July 18, 2016). "Taylor Swift's new music video: A guide to what it all means". CNN. Archived from the original on May 13, 2021. Retrieved February 27, 2021.
Tanzer, Myles (July 18, 2016). "Taylor Swift: "I Would Very Much Like To Be Excluded From This Narrative"". The Fader. Archived from the original on September 1, 2017. Retrieved August 31, 2017.
"It's Taylor Swift vs. Taylor Swift in "Look What You Made Me Do" video". ABC. August 28, 2017. Archived from the original on September 1, 2017. Retrieved August 31, 2017.
Schonfeld, Zach (August 27, 2017). "Taylor Swift's "Look What You Made Me Do" Music Video Is Finally Here". Newsweek. Archived from the original on August 31, 2017. Retrieved August 31, 2017.
Holmes, Sally (August 28, 2017). "All the Old Taylors in Taylor Swift's "Look What You Made Me Do"". Elle. Archived from the original on August 29, 2017. Retrieved August 31, 2017.
Peters, Mitchell (December 2, 2017). "Watch Taylor Swift Perform 'End Game' With Ed Sheeran at Jingle Ball". Billboard. Archived from the original on December 3, 2017. Retrieved December 2, 2017.
Harrington, Jim (December 3, 2017). "Haters, beware – Taylor Swift is back on top again and ready to dominate 2018". The Mercury News. Archived from the original on December 4, 2017. Retrieved December 5, 2017.
Klein, Joshua (December 8, 2017). "Taylor Swift and Backstreet Boys defy fest formula at Jingle Bash". Chicago Tribune. Archived from the original on December 10, 2017. Retrieved December 11, 2017.
Weatherby, Taylor (December 9, 2017). "Charlie Puth Tributes Chris Cornell, Taylor Swift and Ed Sheeran Team Up Again & More From iHeartRadio's Z100 Jingle Ball in NYC". Billboard. Archived from the original on December 10, 2017. Retrieved December 11, 2017.
"Taylor Swift OWNED The #CapitalJBB Stage & Brought All Her Biggest Hits For This Iconic Performance". Capital. December 10, 2017. Archived from the original on December 11, 2017. Retrieved December 11, 2017.
Warner, Denise (May 9, 2018). "Tiffany Haddish Appears on Screen During Taylor Swift's 'Reputation' Tour". Billboard. Archived from the original on May 26, 2018. Retrieved May 14, 2018.
"Here Are All the Songs Taylor Swift Played on the Opening Night of the Reputation Tour". Billboard. US. Archived from the original on May 22, 2018. Retrieved November 29, 2018.
Shafer, Ellise (March 18, 2023). "Taylor Swift Eras Tour: The Full Setlist From Opening Night". Variety. Archived from the original on March 18, 2023. Retrieved March 19, 2023.
Gularte, Alejandra (November 28, 2023). "You Missed These Easter Eggs on Taylor Swift's Eras Tour". Vulture. Retrieved November 20, 2024.
Rowley, Glenn (May 16, 2023). "Here Are the Easter Eggs on Taylor Swift's The Eras Tour". Billboard. Retrieved November 20, 2024.
Lynch, Jason (August 25, 2017). "ABC Is Already Using Taylor Swift's New Song 'Look What You Made Me Do' to Promote TGIT". Adweek. Archived from the original on August 28, 2017. Retrieved August 26, 2017.
Joseph, Andrew (August 28, 2017). "ESPN using Taylor Swift's song for college football ads". USA Today. Archived from the original on August 30, 2017. Retrieved August 28, 2017.
Murder Mystery | Trailer | Netflix (Trailer). Netflix. April 26, 2019. Event occurs at 1 minute 17 seconds. Archived from the original on April 28, 2019. Retrieved April 28, 2019.
Kusiak, Lindsay (January 19, 2022). "Sing 2 Soundtrack Guide: Every Song". ScreenRant. Archived from the original on March 3, 2022. Retrieved March 3, 2022.
Monroe, Jazz (May 25, 2020). "Taylor Swift and Jack Antonoff Team for Mysterious "Look What You Made Me Do" Cover on Killing Eve". Pitchfork. Archived from the original on May 28, 2020. Retrieved May 26, 2020.
Willman, Chris (May 25, 2020). "Taylor Swift's (Apparent) Remake of 'Look What You Made Me Do' with Brother Austin Fires Up Fandom". Variety. Archived from the original on May 26, 2020. Retrieved May 26, 2020.
Sakzewski, Emily (May 26, 2020). "What Taylor Swift's mysterious Killing Eve cover could mean in her feud with Scooter Braun". ABC News. Archived from the original on May 26, 2020. Retrieved May 26, 2020.
Richards, Will (May 25, 2020). "Taylor Swift fans think new 'Killing Eve' cover is her getting back at Scooter Braun". NME. Archived from the original on June 3, 2020. Retrieved May 26, 2020.
Shafer, Ellise (August 23, 2023). "Taylor Swift Unveils 'Look What You Made Me Do (Taylor's Version)' in Prime Video's 'Wilderness' Teaser". Variety. Archived from the original on August 23, 2023. Retrieved August 23, 2023.
"Breaking: The 2017 MTV EMA Nominations Are Here!". Archived from the original on October 6, 2017. Retrieved October 2, 2017.
"Nommés NRJ MUSIC AWARDS 2017" (in French). NRJ Music Awards. Archived from the original on October 1, 2016. Retrieved October 2, 2017.
Estado de Minas (April 2, 2018). "Anitta lidera indicações da primeira edição do MTV Miaw Brasil; confir" (in Portuguese). Uai. Archived from the original on April 3, 2018. Retrieved April 2, 2018.
Daly, Rhian (January 17, 2018). "VO5 NME Awards Nominations List 2018". NME. Archived from the original on January 24, 2018. Retrieved February 11, 2018.
Aniftos, Rania (January 10, 2018). "Rihanna, Ed Sheeran & Bruno Mars Lead iHeartRadio Music Awards 2018 Nominees". Billboard. Archived from the original on June 20, 2018. Retrieved January 17, 2018.
Haring, Bruce (March 25, 2018). "Nickelodeon Kids' Choice Awards 2018 Winners – The Complete List". Deadline. Archived from the original on March 25, 2018. Retrieved March 25, 2018.
"MYXMusicAwards 2016 Winners List". MYX. Archived from the original on August 10, 2016. Retrieved August 6, 2016.
"The 2018 Radio Disney Music Awards Coming to The Dolby Theatre on Friday, June 22, 2018". Broadway World. Archived from the original on June 12, 2018. Retrieved May 25, 2018.
Cohen, Jess (June 13, 2018). "Teen Choice Awards 2018: Avengers: Infinity War, Black Panther and Riverdale Among Top Nominees". E! News. Archived from the original on June 13, 2018. Retrieved June 13, 2018.
"2018 Hito Music Awards". Hit FM (in Chinese). Archived from the original on November 4, 2019. Retrieved July 4, 2020.
"MTV VMAs 2018: See The Full List of Nominees Here". Pitchfork. July 16, 2018. Archived from the original on July 17, 2018. Retrieved April 13, 2019.
"Harry Gregson-Williams and Other Top Songwriters Honored At 2018 BMI London Awards". BMI. October 1, 2018. Archived from the original on October 9, 2018. Retrieved August 11, 2019.
"Revealed: Taylor Swift, Justin Bieber, Drake and BTS rock the Guinness World Records 2019 Edition". Guinness World Records. August 22, 2018. Archived from the original on August 1, 2021. Retrieved February 3, 2021.
"Sting, Imagine Dragons and Martin Bandier Honored at BMI's 67th Annual Pop Awards". BMI. May 16, 2019. Archived from the original on May 15, 2019. Retrieved May 16, 2019.
"Creative Nominees". TEC Awards. Archived from the original on October 23, 2020. Retrieved September 25, 2020.
"Top 20 Anglo Argentina – Del 9 al 15 de Octubre, 2017" (in Spanish). Monitor Latino. October 9, 2017. Archived from the original on October 16, 2017. Retrieved October 16, 2017.
"Taylor Swift – Look What You Made Me Do". ARIA Top 50 Singles. Retrieved September 2, 2017.
"Taylor Swift – Look What You Made Me Do" (in German). Ö3 Austria Top 40. Retrieved September 6, 2017.
"Taylor Swift – Look What You Made Me Do" (in Dutch). Ultratop 50. Retrieved September 8, 2017.
"Taylor Swift – Look What You Made Me Do" (in French). Ultratop 50. Retrieved September 8, 2017.
"Архив класации" (in Bulgarian). PROPHON. Archived from the original on March 21, 2019. Retrieved July 11, 2019.
"Taylor Swift Chart History (Canadian Hot 100)". Billboard. Retrieved September 6, 2017.
"Taylor Swift Chart History (Canada AC)". Billboard. Retrieved September 19, 2017.
"Taylor Swift Chart History (Canada CHR/Top 40)". Billboard. Retrieved September 19, 2017.
"Taylor Swift Chart History (Canada Hot AC)". Billboard. Retrieved September 26, 2017.
Taylor Swift — Look What You Made Me Do. TopHit. Retrieved July 25, 2023.
"Stream Rankings: Semana del 01/09/17 al 07/09/17" (in Spanish). Promúsica Colombia. Archived from the original on June 1, 2022. Retrieved June 1, 2022.
"Top 100 Colombia" (in Spanish). National-Report. Archived from the original on September 26, 2017. Retrieved July 22, 2019.
"Costa Rica General" (in Spanish). Monitor Latino. Archived from the original on November 7, 2017. Retrieved May 10, 2018.
"Croatia ARC TOP 100". HRT. Archived from the original on November 2, 2017. Retrieved October 29, 2017.
"ČNS IFPI" (in Czech). Hitparáda – Radio Top 100 Oficiální. IFPI Czech Republic. Note: Select 39. týden 2017 in the date selector. Retrieved October 2, 2017.
"ČNS IFPI" (in Czech). Hitparáda – Digital Top 100 Oficiální. IFPI Czech Republic. Note: Select 35. týden 2017 in the date selector. Retrieved September 5, 2017.
"Taylor Swift – Look What You Made Me Do". Tracklisten. Retrieved September 6, 2017.
"Top 100 Ecuador" (in Spanish). National-Report. Archived from the original on October 23, 2017. Retrieved June 17, 2020.
"El Salvador General" (in Spanish). Monitor Latino. Archived from the original on May 10, 2018. Retrieved May 10, 2018.
"Taylor Swift Chart History (Euro Digital Song Sales)". Billboard. Retrieved September 6, 2017.
"Taylor Swift: Look What You Made Me Do" (in Finnish). Musiikkituottajat. Retrieved September 4, 2017.
"Le Top de la semaine : Top Singles (téléchargement + streaming) – SNEP (Week 35, 2017)" (in French). Syndicat National de l'Édition Phonographique. Archived from the original on September 24, 2017. Retrieved September 4, 2017.
"Taylor Swift – Look What You Made Me Do" (in German). GfK Entertainment charts. Retrieved September 8, 2017.
"Billboard – Greece Digital Song Sales". Billboard. Archived from the original on August 19, 2017. Retrieved September 2, 2017.
"IFPI Charts". February 21, 2018. Archived from the original on February 21, 2018.
"Honduras General" (in Spanish). Monitor Latino. Archived from the original on May 9, 2018. Retrieved May 9, 2018.
"Archívum – Slágerlisták – MAHASZ" (in Hungarian). Single (track) Top 40 lista. Magyar Hanglemezkiadók Szövetsége. Retrieved September 9, 2017.
"IRMA – Irish Charts". Irish Recorded Music Association. Archived from the original on December 26, 2018. Retrieved September 2, 2017.
"Media Forest Week 37, 2017". Israeli Airplay Chart. Media Forest. Retrieved September 16, 2017.
"Taylor Swift – Look What You Made Me Do". Top Digital Download. Retrieved September 2, 2017.
"Taylor Swift Chart History (Japan Hot 100)". Billboard. Retrieved September 6, 2017.
"The Official Lebanese Top 20 – Taylor Swift". The Official Lebanese Top 20. Archived from the original on September 17, 2016. Retrieved September 25, 2017.
"Taylor Swift Chart History (Luxembourg Digital Song Sales)". Billboard. Retrieved September 2, 2017. [dead link]
"Top 20 Most Streamed International & Domestic Singles in Malaysia : Week 35 (25/8/2017 – 31/8/2017)" (PDF). Recording Industry Association of Malaysia. Archived from the original (PDF) on September 27, 2017. Retrieved September 27, 2017.
"Mexico Airplay: Oct 7, 2017". Billboard. January 2, 2013. Archived from the original on May 23, 2018. Retrieved May 16, 2018.
"Nederlandse Top 40 – week 37, 2017" (in Dutch). Dutch Top 40. Retrieved September 15, 2017.
"Taylor Swift – Look What You Made Me Do" (in Dutch). Single Top 100. Retrieved September 8, 2017.
"Taylor Swift – Look What You Made Me Do". VG-lista. Retrieved September 9, 2017.
"Panamá Top 20 - Del 09 al 15 de Octubre, 2017". Monitor Latino. Archived from the original on June 26, 2020. Retrieved April 18, 2020.
"Portuguese Charts – Singles Top 20". portuguesecharts.com. Archived from the original on September 8, 2017. Retrieved November 7, 2017.
"2, 2017 Russia Airplay Chart for October 2, 2017." TopHit. Retrieved April 19, 2021.
"Airplay 100 – 24 septembrie 2017" (in Romanian). Kiss FM. September 24, 2017. Archived from the original on February 21, 2019. Retrieved February 20, 2019.
"Official Scottish Singles Sales Chart Top 100". Official Charts Company. Retrieved September 1, 2017.
"ČNS IFPI" (in Slovak). Hitparáda – Singles Digital Top 100 Oficiálna. IFPI Czech Republic. Note: Select 35. týden 2017 in the date selector. Retrieved September 5, 2017.
"SloTop50: Slovenian official singles weekly chart" (in Slovenian). SloTop50. Archived from the original on August 31, 2018. Retrieved August 30, 2018.
"Digital Chart" (in Korean). Circle Music Chart. Archived from the original on June 2, 2023. Retrieved June 2, 2023.
"Taylor Swift – Look What You Made Me Do" Canciones Top 50. Retrieved September 14, 2017.
"Taylor Swift – Look What You Made Me Do". Singles Top 100. Retrieved September 8, 2017.
"Taylor Swift – Look What You Made Me Do". Swiss Singles Chart. Retrieved September 4, 2017.
"Official Singles Chart Top 100". Official Charts Company. Retrieved September 2, 2017.
"Uruguay General" (in Spanish). Monitor Latino. Archived from the original on November 13, 2017. Retrieved May 14, 2018.
"Taylor Swift Chart History (Hot 100)". Billboard. Retrieved August 29, 2017.
"Taylor Swift Chart History (Adult Contemporary)". Billboard. Retrieved September 6, 2017.
"Taylor Swift Chart History (Adult Pop Songs)". Billboard. Retrieved September 26, 2017.
"Taylor Swift Chart History (Dance Club Songs)". Billboard. Retrieved October 17, 2017.
"Taylor Swift Chart History (Dance Mix/Show Airplay)". Billboard. Retrieved October 28, 2017.
"Taylor Swift Chart History (Pop Songs)". Billboard. Retrieved October 17, 2017.
"Taylor Swift Chart History (Rhythmic)". Billboard. Retrieved September 26, 2017.
"Top 20 Anglo Venezuela" (in Spanish). Monitor Latino. September 25, 2017. Archived from the original on December 4, 2017. Retrieved September 25, 2017.
"Top 100 Venezuela". National-Report. Archived from the original on January 12, 2018. Retrieved February 16, 2023.
"RIAS Top Charts Week 10 (1 - 7 Mar 2024)". RIAS. Archived from the original on March 12, 2024. Retrieved March 12, 2024.
"ARIA End of Year Singles 2017". Australian Recording Industry Association. Archived from the original on June 8, 2019. Retrieved January 5, 2018.
"Ö3 Austria Top 40 – Single-Charts 2017". oe3.orf.at. Archived from the original on December 30, 2017. Retrieved December 29, 2017.
"Top 200 faixas em streaming - 2017" (PDF) (in Portuguese). Pro-Música Brasil. Archived (PDF) from the original on March 6, 2020. Retrieved June 2, 2022.
"Canadian Hot 100 – Year-End 2017". Billboard. January 2, 2013. Archived from the original on December 17, 2017. Retrieved September 18, 2019.
"Costa Rica – 2017 Year-End charts". Monitor Latino. December 18, 2017. Archived from the original on January 15, 2020. Retrieved July 29, 2020.
"El Salvador – 2017 Year-End charts". Monitor Latino. December 18, 2017. Archived from the original on December 15, 2017. Retrieved July 28, 2020.
"Top 100 Single-Jahrescharts". GfK Entertainment (in German). offiziellecharts.de. Archived from the original on June 8, 2019. Retrieved December 29, 2017.
"Top 100 Anual 2017 Honduras" (in Spanish). Monitor Latino. Archived from the original on August 7, 2020. Retrieved July 28, 2020.
"Single Top 100 – eladási darabszám alapján – 2017". Mahasz. Archived from the original on February 4, 2020. Retrieved February 17, 2018.
"Stream Top 100 – 2017". Mahasz. Archived from the original on February 4, 2020. Retrieved February 17, 2018.
"Top 100-Jaaroverzicht van 2017". Dutch Top 40. Archived from the original on July 29, 2019. Retrieved October 21, 2019.
"Top 100 Anual 2017 Nicaragua" (in Spanish). Monitor Latino. Archived from the original on May 25, 2018. Retrieved May 30, 2018.
"Paraguay – 2017 Year-End Charts" (in Spanish). Monitor Latino. December 18, 2017. Archived from the original on August 7, 2020. Retrieved July 28, 2020.
"Top AFP - Audiogest - Top 100 Singles 2017" (PDF) (in Portuguese). Associação Fonográfica Portuguesa. Archived (PDF) from the original on January 1, 2021. Retrieved August 16, 2020.
"End of Year Singles Chart Top 100 – 2017". Official Charts Company. Archived from the original on February 12, 2016. Retrieved January 10, 2018.
"Hot 100 Songs – Year-End 2017". Billboard. January 2, 2013. Archived from the original on December 16, 2017. Retrieved December 12, 2017.
"Adult Pop Songs - Year End 2017". Billboard. January 2, 2013. Archived from the original on June 30, 2018. Retrieved September 18, 2019.
"Pop Songs - Year End 2017". Billboard. January 2, 2013. Archived from the original on June 30, 2018. Retrieved September 18, 2019.
"국내 대표 음악 차트 가온차트!" (in Korean). Gaon. Archived from the original on April 1, 2019. Retrieved January 13, 2019.
"Austrian single certifications – Taylor Swift – Look What You Made Me Do" (in German). IFPI Austria. Retrieved May 29, 2024.
"Ultratop − Goud en Platina – singles 2017". Ultratop. Hung Medien. Retrieved October 11, 2021.
"Brazilian single certifications – Taylor Swift – Look What You Made Me Do" (in Portuguese). Pro-Música Brasil. Retrieved July 23, 2024.
"Danish single certifications – Taylor Swift – Look What You Made Me Do". IFPI Danmark. Retrieved April 1, 2020.
"French single certifications – Taylor Swift – Look What You Made Me Do" (in French). Syndicat National de l'Édition Phonographique. Retrieved March 13, 2018.
"Gold-/Platin-Datenbank (Taylor Swift; 'Look What You Made Me Do')" (in German). Bundesverband Musikindustrie. Retrieved December 13, 2017.
"Italian single certifications – Taylor Swift – Look What You Made Me Do" (in Italian). Federazione Industria Musicale Italiana. Retrieved November 13, 2017.
"New Zealand single certifications – Taylor Swift – Look What You Made Me Do". Radioscope. Retrieved December 19, 2024. Type Look What You Made Me Do in the "Search:" field.
"Norwegian single certifications – Taylor Swift – Look What You Made Me Do" (in Norwegian). IFPI Norway. Retrieved October 29, 2020.
"Wyróżnienia – Platynowe płyty CD - Archiwum - Przyznane w 2020 roku" (in Polish). Polish Society of the Phonographic Industry. Retrieved May 7, 2020.
"Portuguese single certifications – Taylor Swift – Look What You Made Me Do" (PDF) (in Portuguese). Associação Fonográfica Portuguesa. Retrieved May 23, 2022.
"Spanish single certifications – Taylor Swift – Look What You Made Me Do". El portal de Música. Productores de Música de España. Retrieved January 12, 2024.
"Sverigetopplistan – Taylor Swift" (in Swedish). Sverigetopplistan.
"British single certifications – Taylor Swift – Look What You Made Me Do". British Phonographic Industry. Retrieved October 13, 2023.
"American single certifications – Taylor Swift – Look What You Made Me Do". Recording Industry Association of America. Retrieved July 23, 2018.

    "'Look What You Made Me Do' – Taylor Swift" (in Italian). Radio Airplay. Archived from the original on August 27, 2017. Retrieved August 25, 2017.

Cited literature

    Bengtsson, Linda Ryan; Edlom, Jessica (2023). "Commodifying participation through choreographed engagement: the Taylor Swift case". Arts and the Market. 13 (2): 65–79. doi:10.1108/AAM-07-2022-0034.
    McNutt, Myles (2020). "From 'Mine' to 'Ours': Gendered Hierarchies of Authorship and the Limits of Taylor Swift's Paratextual Feminism". Communication, Culture and Critique. 13 (1): 72–91. doi:10.1093/ccc/tcz042.
    Wilkinson, Maryn (2017). "Taylor Swift: the hardest working, zaniest girl in show business". Celebrity Studies. 10 (3): 441–444. doi:10.1080/19392397.2019.1630160.
    Zaleski, Annie (2024). "The Reputation Era". Taylor Swift: The Stories Behind the Songs. Thunder Bay Press. pp. 132–147. ISBN 978-1-6672-0845-9.

    vte

Taylor Swift songs

    Singles discography

Taylor Swift	

    "Tim McGraw" "Picture to Burn" "Teardrops on My Guitar" "A Place in This World" "Should've Said No" "Our Song"

Fearless	

    "Fearless" "Fifteen" "Love Story" "Hey Stephen" "White Horse" "You Belong with Me" "Breathe" "You're Not Sorry" "The Way I Loved You" "Forever & Always" "The Best Day" "Change"

Taylor's Version	

    "You All Over Me" "Mr. Perfectly Fine" "That's When"

Speak Now	

    "Mine" "Sparks Fly" "Back to December" "Speak Now" "Dear John" "Mean" "The Story of Us" "Never Grow Up" "Enchanted" "Better than Revenge" "Innocent" "Haunted" "Long Live" "Ours"

Taylor's Version	

    "Electric Touch" "When Emma Falls in Love" "I Can See You" "Castles Crumbling"

Red	

    "State of Grace" "Red" "Treacherous" "I Knew You Were Trouble" "All Too Well" "22" "We Are Never Ever Getting Back Together" "The Last Time" "Holy Ground" "Everything Has Changed" "Begin Again"

Taylor's Version	

    "Nothing New" "Message in a Bottle" "I Bet You Think About Me" "Forever Winter"

1989	

    "Welcome to New York" "Blank Space" "Style" "Out of the Woods" "All You Had to Do Was Stay" "Shake It Off" "I Wish You Would" "Bad Blood" "Wildest Dreams" "How You Get the Girl" "This Love" "I Know Places" "Clean" "You Are in Love" "New Romantics"

Taylor's Version	

    "'Slut!'" "Say Don't Go" "Now That We Don't Talk" "Suburban Legends" "Is It Over Now?"

Reputation	

    "...Ready for It?" "End Game" "I Did Something Bad" "Don't Blame Me" "Delicate" "Look What You Made Me Do" "So It Goes..." "Gorgeous" "Getaway Car" "Dress" "Call It What You Want" "New Year's Day"

Lover	

    "I Forgot That You Existed" "Cruel Summer" "Lover" "The Man" "The Archer" "Miss Americana & the Heartbreak Prince" "Paper Rings" "Cornelia Street" "Death by a Thousand Cuts" "London Boy" "Soon You'll Get Better" "False God" "You Need to Calm Down" "Me!"

Folklore	

    "The 1" "Cardigan" "The Last Great American Dynasty" "Exile" "My Tears Ricochet" "Mirrorball" "Seven" "August" "This Is Me Trying" "Illicit Affairs" "Invisible String" "Mad Woman" "Epiphany" "Betty" "Peace" "Hoax" "The Lakes"

Evermore	

    "Willow" "Champagne Problems" "'Tis the Damn Season" "Tolerate It" "No Body, No Crime" "Happiness" "Dorothea" "Coney Island" "Long Story Short" "Marjorie"

Midnights	

    "Lavender Haze" "Maroon" "Anti-Hero" "Snow on the Beach" "You're on Your Own, Kid" "Midnight Rain" "Question...?" "Vigilante Shit" "Bejeweled" "Labyrinth" "Karma" "Sweet Nothing" "Mastermind" "Hits Different" "Bigger Than the Whole Sky" "Would've, Could've, Should've" "You're Losing Me"

The Tortured Poets
Department	

    "Fortnight" "The Tortured Poets Department "My Boy Only Breaks His Favorite Toys" "Down Bad" "So Long, London" "But Daddy I Love Him" "Fresh Out the Slammer" "Florida!!!" "Guilty as Sin?" "Who's Afraid of Little Old Me?" "Loml" "I Can Do It with a Broken Heart" "The Smallest Man Who Ever Lived" "The Alchemy" "Clara Bow" "The Black Dog" "So High School" "Thank You Aimee"

Soundtrack songs	

    "Crazier" "Today Was a Fairytale" "Safe & Sound" "Eyes Open" "Sweeter than Fiction" "I Don't Wanna Live Forever" "Beautiful Ghosts" "Only the Young" "Carolina"

Featured songs	

    "Two Is Better Than One" "Half of My Heart" "Both of Us" "Babe" "Gasoline" "Renegade" "The Joker and the Queen" "The Alcott" "Us"

Other songs	

    "Best Days of Your Life" "Ronan" "Highway Don't Care" "Better Man" "Christmas Tree Farm" "All of the Girls You Loved Before"

    Category

Authority control databases Edit this at Wikidata	

    MusicBrainz workMusicBrainz release group

Categories:

    2017 singles2017 songsTaylor Swift songsSongs written by Taylor SwiftSongs written by Jack AntonoffBig Machine Records singlesAmerican dance-pop songsElectropop songsProgressive pop songsMusic videos directed by Joseph KahnSong recordings produced by Taylor SwiftSong recordings produced by Jack AntonoffDiss tracksSongs about revengeBillboard Hot 100 number-one singlesCanadian Hot 100 number-one singlesIrish Singles Chart number-one singlesUK singles chart number-one singlesNumber-one singles in AustraliaNumber-one singles in GreeceNumber-one singles in IsraelNumber-one singles in MalaysiaNumber-one singles in New ZealandNumber-one singles in ScotlandNumber-one singles in the PhilippinesSongs written by Richard FairbrassSongs written by Fred FairbrassSongs written by Rob Manzoli

    This page was last edited on 3 January 2025, at 17:33 (UTC).
    Text is available under the Creative Commons Attribution-ShareAlike 4.0 License; additional terms may apply. By using this site, you agree to the Terms of Use and Privacy Policy. Wikipedia® is a registered trademark of the Wikimedia Foundation, Inc., a non-profit organization.

    Privacy policy
    About Wikipedia
    Disclaimers
    Contact Wikipedia
    Code of Conduct
    Developers
    Statistics
    Cookie statement
    Mobile view

    Wikimedia Foundation
    Powered by MediaWiki



Wikipedia The Free Encyclopedia

    Donate
    Create account
    Log in

Contents
(Top)
Background and production

Lyrics and music

Critical reception

Release and commercial performance

Music video

Accolades

Live performances

Covers and other usage

Personnel

Charts

    Weekly charts
    Year-end charts
    All-time charts

Certifications

"I Knew You Were Trouble (Taylor's Version)"

    Personnel
    Charts
        Weekly charts
        Year-end charts
    Certifications

See also

Footnotes

References

        Source

I Knew You Were Trouble

    Article
    Talk

    Read
    Edit
    View history

Tools

Appearance
Text

    Small
    Standard
    Large

Width

    Standard
    Wide

Color (beta)

    Automatic
    Light
    Dark

This is a good article. Click here for more information.
From Wikipedia, the free encyclopedia
"I Knew You Were Trouble"
A photo of Swift in a white collar shirt holding her sunglasses while staring upwards, displaying the song title and her name
Single by Taylor Swift
from the album Red
Released	November 27, 2012
Studio	

    MXM (Stockholm)
    Conway (Los Angeles)

Genre	

    Dance-pop pop rock

Length	3:39
Label	Big Machine
Songwriter(s)	

    Taylor Swift Max Martin Shellback

Producer(s)	

    Max Martin Shellback

Taylor Swift singles chronology
"Begin Again"
(2012) 	"I Knew You Were Trouble"
(2012) 	"22"
(2013)
Music video
"I Knew You Were Trouble" on YouTube

"I Knew You Were Trouble"[note 1] is a song by the American singer-songwriter Taylor Swift from her fourth studio album, Red (2012). She wrote the song with its producers, Max Martin and Shellback. Instrumented by electric guitars and synthesizers, "I Knew You Were Trouble" has a production that incorporates pop subgenres. Its refrain is accentuated by a dubstep wobble and Swift's distorted vocals; music critics found the dubstep production a radical departure from her previous country pop sounds. The lyrics are about a narrator's self-blame after a toxic relationship upon recognizing the warning signs in hindsight.

Big Machine, in partnership with Republic Records, released "I Knew You Were Trouble" to US pop radio on November 27, 2012, as the second pop single and the third overall from Red. The song peaked within the top 10 on record charts and received multi-platinum certifications in Australia, Austria, Canada, New Zealand, and the UK. In the US, the single peaked at number two on the Billboard Hot 100 and spent seven weeks at number one on the Pop Songs chart. Its success on pop radio inspired Swift to recalibrate her artistic identity from country for pop on her 2014 album 1989. Initial reviews were divided: positive comments found the production bold, but criticisms deemed it derivative. Retrospective opinions have regarded "I Knew You Were Trouble" as one of Swift's career-defining singles.

The music video for "I Knew You Were Trouble" premiered on MTV on December 13, 2012. Directed by Anthony Mandler, it depicts Swift with an unfaithful man and ending up alone in a desert. The video won an MTV Video Music Award for Best Female Video and the Phenomenon Award at the YouTube Music Awards in 2013. Swift performed the song at awards shows including the American Music Awards, the ARIA Music Awards, and the Brit Awards. She included "I Knew You Were Trouble" on the set lists of three of her world tours: the Red Tour (2013–2014), the 1989 World Tour (2015), and the Eras Tour (2023–2024). After a 2019 dispute over the ownership of Swift's back catalog, she re-recorded the song as "I Knew You Were Trouble (Taylor's Version)" for her 2021 album Red (Taylor's Version).
Background and production

On her fourth studio album, Red (2012), Taylor Swift aimed to experiment with musical styles other than the country pop sound that had defined her artistry.[2] To do so, she worked with producers located outside her career base in Nashville, Tennessee.[3][4] One such producer is Max Martin, whose production inspired Swift by "how [it] can just land a chorus".[5][6] Swift met Martin and Shellback in Los Angeles, and the three wrote three songs on Red, including "I Knew You Were Trouble".[7][8]

Swift developed the basic melody of "I Knew You Were Trouble" on piano. She asked Martin and Shellback to help with the production, and they incorporated elements of dubstep, a genre that was popular in UK clubs at the time. Although Swift had been familiarized with dubstep through the music that Ed Sheeran introduced to her, she was not aiming to embark on specific trends and instead wanted the sound to convey the chaotic emotions of the lyrics.[9] Deeming it the boldest musical direction of the album, Swift thought that the audience would be "freaked out over" when they listened to it.[10][11]

"I Knew You Were Trouble" was recorded by Michael Ilbert at MXM Studios in Stockholm, Sweden, and by Sam Holland at Conway Recording Studios in Los Angeles, California. It was mixed by Şerban Ghenea at MixStar Studios in Virginia Beach, Virginia, and mastered by Tom Coyne at Sterling Sound Studio in New York City. Martin and Shellback both produced the track and played keyboards in it; the latter also programmed the track and played acoustic guitar, electric guitar, and bass guitar.[12]
Lyrics and music
"I Knew You Were Trouble"
Duration: 23 seconds.0:23
"I Knew You Were Trouble" is a dance-pop and pop rock song with a dubstep refrain. Critics noted the dubstep experimentation as a radical sonic change from Swift's previous country pop songs.
Problems playing this file? See media help.

"I Knew You Were Trouble" is about being involved with someone who is irresistible but flawed: Swift's narrator blames herself for falling in love with him upon recognizing the warning signs in hindsight.[9][13] The song uses the verse–chorus form with an added post-chorus.[14] Music critics described "I Knew You Were Trouble" as a dance-pop,[15][16] pop rock,[17][18] and teen pop song.[19] It features bass guitar, electric guitar, and keyboard.[15] The melody displays influences of dancehall, while Swift's processed vocals were influenced by R&B.[20] The dubstep refrain includes a wobble, synthesizers, and Swift's distorted vocals.[15][18] The instrumental halts at the bridge, where Swift contemplates on her past relationship: "You never loved me, or her, or anyone, or anything."[21]

Critics considered the dubstep experimentation on "I Knew You Were Trouble" a significant departure from Swift's country pop beginnings.[13][17][21] While describing how the song's style felt "sudden" and "unexpected" when compared to other tracks on Red, the musicologist James Perone believed that it was "logical" for first-time listeners to react in surprise upon hearing "I Knew You Were Trouble" if they were familiar with Swift's work prior to its release.[22] Jon Caramanica of The New York Times commented that the dubstep wobble was a "wrecking ball" that shifted the dynamic of not only "the song but also of Ms. Swift's career".[23] Randall Roberts from the Los Angeles Times remarked that although dubstep had been popularized by DJs such as Zedd and Skrillex, "I Knew You Were Trouble" generated much discussion because it introduced the genre to a wider audience of mainstream pop, which had been "sonically conservative for the past half-decade".[24] In Pitchfork, Brad Nelson commented that the production was "sharp as [Swift's] lyrics".[18]
Critical reception

The dubstep experimentation divided contemporary critics.[24][25] James Reed from The Boston Globe wrote that "I Knew You Were Trouble" and the other tracks produced by Martin and Shellback were unoriginal.[26] Amanda Dobbins from Vulture felt the dubstep sound was not innovative, but praised the song as "yet another plucky, vowel-laden Taylor Swift breakup jam".[27] In a Red album review for The Washington Post, Allison Stewart criticized the production as "gratuitous and weird" which overshadowed Swift's lyrics.[28] In defense of Swift, Randall Roberts from the Los Angeles Times said it was "unfair to criticize a 22-year-old for adapting with the times". Though Roberts acknowledged that critics could dismiss the refrain's bass drop as conceit, it was justifiable for Swift—whom he considered a leading pop star—to experiment with mainstream trends.[24]

In positive reviews, Jon Caramanica from The New York Times[23] and Chris Willman from The Hollywood Reporter praised the song for exhibiting Swift's versatility beyond country.[21] Slant Magazine's Jonathan Keefe praised "I Knew You Were Trouble" as one of Red's best tracks because "the production is creative and contemporary in ways that are in service to Swift's songwriting".[17] In a review for Spin, Mark Hogan praised Swift's songcraft and remarked that although the dubstep experimentation initially came off as unoriginal, it "ultimately gets absorbed into [Swift's] own aesthetic".[15]

The song featured on 2012 year-end lists by Spin (34th)[29] and The Village Voice's Pazz & Jop critics' poll (59th).[30] Retrospective reviews of "I Knew You Were Trouble" have been generally positive. Hannah Mylrea from NME and Alexis Petridis from The Guardian considered the single a bold artistic statement for Swift, ranking it among the best songs of her catalog.[25][31] In a 2021 retrospective review, Laura Snapes from The Guardian commented that the song was "the rare pop-EDM crossover" that stood the test of time.[32]
Release and commercial performance

Swift premiered one Red album track each week on Good Morning America, from September 24 until the album's release date of October 22, 2012, as part of a four-week countdown.[33] "I Knew You Were Trouble" was the third song that Swift premiered, on October 8, 2012.[34][35] The day after the Good Morning America premiere, Big Machine Records released the song onto the iTunes Store for digital download.[36][15] Big Machine in partnership with Republic Records released "I Knew You Were Trouble" to US pop radio on November 27, 2012, as an official single.[37] It was the second pop radio single from Red, following "We Are Never Ever Getting Back Together".[38][39] A limited CD single edition featuring fan-exclusive merchandise was available through Swift's official website on December 13, 2012.[40][41] "I Knew You Were Trouble" was released as a radio single in the U.K. on December 9, 2012,[42] and in Italy on January 11, 2013.[43]

On November 3, 2014, Swift removed her entire catalog from on-demand streaming platform Spotify, arguing that their ad-supported free service undermined the platform's premium service, which provides higher royalties for songwriters.[44] In December 2015, the media reported that "I Knew You Were Trouble" had been re-delivered to Spotify, but its credit was mistakenly given to Welsh band Lostprophets and lead singer Ian Watkins. The song was removed from the site after three days.[45] Swift re-added her entire catalog on Spotify in June 2017.[46]

In the US, after its digital release, "I Knew You Were Trouble" debuted at number three on the Billboard Hot 100 and number one on the Digital Songs chart with 416,000 copies sold during the first week. It was Swift's eleventh song to debut in the top ten of the Hot 100. Together with Red's lead single "We Are Never Ever Getting Back Together", it made Swift the first artist in digital history to have two 400,000 digital sales opening weeks.[note 2] After its radio release, the single returned to the top ten on the Billboard Hot 100 and the number-one position on the Digital Songs chart in December 2012 – January 2013.[48][49] Buoyed by strong digital sales, "I Knew You Were Trouble" reached its peak at number two on the Hot 100 chart dated January 12, 2013, behind Bruno Mars' "Locked Out of Heaven" (2012).[38]

The single was Swift's first number-one entry on Adult Top 40.[50] Despite not being released to country radio, the single debuted and stayed for one week at number 55 on the Country Airplay chart in April 2013, resulted from 33 unsolicited plays from Los Angeles radio station KKGO.[51] "I Knew You Were Trouble" spent seven weeks atop the Mainstream Top 40, a chart monitoring pop radio in the US.[52] It was her second Mainstream Top 40 number one, following 2008's "Love Story", and became her single with the most weeks atop the chart.[53] The single's success on pop radio prompted Swift to abandon country and transition to pop on her next studio album, 1989 (2014), which was executive-produced by Swift and Martin.[6][11] By July 2019, "I Knew You Were Trouble" had sold 5.42 million digital copies in the US.[54] The Recording Industry Association of America (RIAA) certified the single seven times platinum for surpassing seven million units based on sales and on-demand streaming.[55]

In Canada, the single peaked at number two on the Canadian Hot 100 and was certified five times platinum by Music Canada (MC).[56] "I Knew You Were Trouble" charted in the top ten on record charts of European countries, peaking at number one in the Czech Republic,[57] number three in Denmark,[58] number four in Ireland,[59] number six in Austria[60] and Russia,[61] number eight in the Commonwealth of Independent States[62] and Switzerland,[63] number nine in Germany,[64] and number ten in Belgian Flanders[65] and Finland.[66] The song received platinum certifications in Germany and Switzerland.[67][68] In the UK, "I Knew You Were Trouble" peaked at number two on the singles chart and was certified triple platinum by the British Phonographic Industry (BPI).[69][70] It peaked at number three and was certified multi-platinum in Australia (six times platinum)[71] and New Zealand (double platinum).[72]
Music video
A man performing onstage with a guitar
Reeve Carney (pictured in 2010) portrays Swift's love interest in the music video.

Anthony Mandler directed the music video for "I Knew You Were Trouble".[73] Shot in Los Angeles over two days, the video stars Reeve Carney as Swift's love interest.[74][75] In the video, Swift wears a pink ombré hairstyle, a ripped tee-shirt, and skinny jeans.[76][77] Marie Claire commented that this "edgy" look coincided with her much publicized relationship with English singer Harry Styles, which signified her outgrowing "good girl" public image.[78]

Swift summarized the video's narrative: "I wanted to tell the story of a girl who falls into a world that's too fast for her, and suffers the consequences."[77] The video begins with Swift waking up in a desert filled with trash and debris from a concert the night before, intertwined with flashbacks of her and her love interest.[79] Swift delivers a monologue reflecting on the past relationship, concluding: "I think that the worst part of it all wasn't losing him. It was losing me."[78] As the song begins, Swift and the love interest are seen sharing intimate moments together. He exhibits behaviors that are unreliable, engaging in bar fights and making out with other girls in a rave.[80] The video concludes with Swift alone in the same desert from the beginning.[81]

Media publications commented on the video's narrative and style. Spin's Chris Martins and Vulture's Amanda Dobbins noted similarities—the desert settings, the "bad boy" love interests, the partying scenes—to Lana Del Rey's 2012 video for "Ride",[73][81] while Rolling Stone compared the downward spiral of Swift's relationship to that portrayed in Rihanna's 2011 video for "We Found Love".[79] Comments by Wendy Geller from Yahoo!,[77] Melinda Newman from Uproxx,[76] and Rachel Brodsky from MTV focused on the video's dark narrative, which depicted a new aspect of Swift's artistry.[80] Martins was not enthusiastic, calling the video unoriginal.[73] A remix of "I Knew You Were Trouble" containing sounds of a screaming goat went viral, resulting in internet memes and boosting the video's popularity.[82][83]
Accolades

"I Knew You Were Trouble" was one of the award-winning songs at the 2014 BMI Awards.[84] It was one of the "Most Performed Songs" awarded at the 2014 ASCAP Awards, in honor of songwriters and producers.[85] The song won Song of the Year at the 2013 Radio Disney Music Awards.[86] At the 2013 MTV Video Music Awards, "I Knew You Were Trouble" won Best Female Video and was nominated for Video of the Year; it was Swift's second win in the category following "You Belong with Me in 2009.[87] It also won YouTube Phenomenon at the inaugural YouTube Music Awards in 2013.[88] The song received nominations at popularity-catered awards ceremonies including Nickelodeon Australian Kids' Choice Awards,[89] Teen Choice Awards,[90] and Nickelodeon Kids' Choice Awards.[91]
Live performances
Swift in a white ball gown singing surrounded by male dancers
Swift performing "I Knew You Were Trouble" on the Red Tour

Swift performed "I Knew You Were Trouble" for the first time at the 2012 American Music Awards, held at Nokia Theatre L.A. Live on November 18, 2012.[92] She embarked on a promotional tour for Red in Australia and performed the song on Today[93] and the ARIA Music Awards.[94] During Red's promotional campaign in the US, Swift included "I Knew You Were Trouble" in her performances at KIIS-FM Jingle Ball on December 1,[95] Z100 Jingle Ball Concert at Madison Square Garden on December 7,[96] and on Dick Clark's New Year's Rockin' Eve at Times Square on December 31, 2012.[97]

On January 18, 2013, following an appearance at the NRJ Music Awards, Swift held a private concert in Paris, where she included "I Knew You Were Trouble" in the set list.[98] She also made live appearances in the U.K., performing the song at the 33rd Brit Awards on February 20,[99] and on The Graham Norton Show on February 23, 2013.[100] "I Knew You Were Trouble" was part of the regular set list of the Red Tour (2013), a world tour Swift embarked on to promote the album.[101] For both BRIT Awards performance and The Red Tour concerts, Swift first performed in a white-and-gold gown with masquerade dancers, and midway changed the costume to black romper and high heels.[102][103]

"I Knew You Were Trouble" is a recurring song included in many of Swift's live performances outside promotion of Red. She performed the song at the Victoria's Secret Fashion Show 2013, broadcast by CBS on December 10, 2013.[104] During the promotion of her 2014 album 1989, Swift performed the song at the iHeartRadio Music Festival on September 19,[105] the We Can Survive benefit concert at the Hollywood Bowl on October 24,[106] and the Jingle Ball Tour 2014 on December 5, 2014.[107]

During the concerts of the 1989 World Tour (2015), she included an industrial rock-oriented version of "I Knew You Were Trouble" in the set lists.[108][109] An acoustic version of "I Knew You Were Trouble" was a "surprise song" Swift performed at the first concert in Manchester, England, and the concert in Perth, Australia, as part of her Reputation Stadium Tour (2018).[110] During the promotion of her 2019 album Lover, Swift again performed the song at the Wango Tango festival on June 1,[111] the Amazon Prime Day concert on July 10,[112] and the City of Lover one-off concert in Paris on September 9, 2019.[113] At the 2019 American Music Awards, where she was honored as the Artist of the Decade, Swift performed "I Knew You Were Trouble" as part of a medley of her biggest hits.[114] Swift included the song on the set list of the Eras Tour (2023–2024).[115]
Covers and other usage

The song was covered by American metalcore band We Came As Romans as part of Fearless Records' Punk Goes Pop Vol. 6.[116] American singer Sabrina Carpenter recorded a stripped-down cover of the song as part of the Spotify Singles series.[117]
Personnel

Credits are adapted from the liner notes of Red.[12]

    Taylor Swift – lead vocals, backing vocals, writer
    Max Martin – producer, writer, keyboards
    Shellback – producer, writer, acoustic guitar, electric guitar, bass, keyboards, programming
    Tom Coyne – mastering
    Eric Eylands – assistant
    Şerban Ghenea – mixing
    John Hanes – engineering
    Sam Holland – recording
    Michael Ilbert – recording
    Tim Roberts – assistant
    JoAnn Tominaga – production co-ordinator

Charts
Weekly charts
Weekly charts performance for "I Knew You Were Trouble" Chart (2012–2013) 	Peak
position
Australia (ARIA)[118] 	3
Austria (Ö3 Austria Top 40)[60] 	6
Belgium (Ultratop 50 Flanders)[65] 	10
Belgium (Ultratop 50 Wallonia)[119] 	32
Brazil (Billboard Brasil Hot 100)[120] 	24
Bulgaria (IFPI)[121] 	29
Canada (Canadian Hot 100)[122] 	2
Canada AC (Billboard)[123] 	5
Canada CHR/Top 40 (Billboard)[124] 	1
Canada Hot AC (Billboard)[125] 	1
CIS Airplay (TopHit)[62] 	8
Croatia (HRT)[126] 	1
Czech Republic (Rádio – Top 100)[57] 	1
Denmark (Tracklisten)[58] 	3
Euro Digital Song Sales (Billboard)[127] 	3
Finland Download (Latauslista)[66] 	10
France (SNEP)[128] 	14
Germany (GfK)[64] 	9
Hungary (Rádiós Top 40)[129] 	13
Ireland (IRMA)[59] 	4
Israel (Media Forest)[130] 	2
Japan (Japan Hot 100)[131] 	51
Japan Adult Contemporary (Billboard)[132] 	22
Lebanon (Lebanese Top 20)[133] 	3
Luxembourg Digital Song Sales (Billboard)[134] 	6
Mexico Anglo (Monitor Latino)[135] 	15
Netherlands (Dutch Top 40)[136] 	19
Netherlands (Single Top 100)[137] 	27
New Zealand (Recorded Music NZ)[138] 	3
Russia Airplay (TopHit)[61] 	6
Scotland (OCC)[139] 	2
Slovakia (Rádio Top 100)[140] 	17
Spain (PROMUSICAE)[141] 	17
Sweden (Sverigetopplistan)[142] 	37
Switzerland (Schweizer Hitparade)[63] 	8
Turkey (Number One Top 20)[143] 	6
UK Singles (OCC)[69] 	2
Ukraine Airplay (TopHit)[144] 	44
US Billboard Hot 100[145] 	2
US Adult Contemporary (Billboard)[146] 	5
US Adult Pop Airplay (Billboard)[147] 	1
US Country Airplay (Billboard)[148] 	55
US Dance/Mix Show Airplay (Billboard)[149] 	18
US Pop Airplay (Billboard)[150] 	1
US Rhythmic (Billboard)[151] 	21
	
Year-end charts
2012 year-end chart performance for "I Knew You Were Trouble" Chart (2012) 	Position
Australia (ARIA)[152] 	27
New Zealand (Recorded Music NZ)[153] 	33
UK Singles (Official Charts Company)[154] 	102
2013 year-end chart performance for "I Knew You Were Trouble" Chart (2013) 	Position
Australia (ARIA)[155] 	55
Austria (Ö3 Austria Top 40)[156] 	34
Belgium (Ultratop Flanders)[157] 	51
Belgium (Ultratop Wallonia)[158] 	81
Canada (Canadian Hot 100)[159] 	12
Denmark (Tracklisten)[160] 	23
France (SNEP)[161] 	43
Germany (Official German Charts)[162] 	57
Hungary (Rádiós Top 40)[163] 	94
Ireland (IRMA)[164] 	17
Netherlands (Dutch Top 40)[165] 	72
New Zealand (Recorded Music NZ)[166] 	31
Russia Airplay (TopHit)[167] 	32
Switzerland (Schweizer Hitparade)[168] 	34
Ukraine Airplay (TopHit)[169] 	86
UK Singles (Official Charts Company)[170] 	17
US Billboard Hot 100[171] 	16
US Adult Contemporary (Billboard)[172] 	17
US Adult Top 40 (Billboard)[173] 	18
US Mainstream Top 40 (Billboard)[174] 	3
US Radio Songs (Billboard)[175] 	9
All-time charts
1992–2017 all-time chart performance for "I Knew You Were Trouble" Chart (1992–2017) 	Position
US Mainstream Top 40 (Billboard)[176] 	49

Certifications
Certifications for "I Knew You Were Trouble" Region 	Certification 	Certified units/sales
Australia (ARIA)[71] 	11× Platinum 	770,000‡
Austria (IFPI Austria)[177] 	2× Platinum 	60,000*
Belgium (BEA)[178] 	Gold 	15,000*
Brazil (Pro-Música Brasil)[179] 	Diamond 	160,000‡
Canada (Music Canada)[56] 	5× Platinum 	400,000*
Denmark (IFPI Danmark)[180] 	Platinum 	90,000‡
Germany (BVMI)[67] 	Platinum 	300,000‡
Italy (FIMI)[181] 	Gold 	25,000‡
Japan (RIAJ)[182] 	Gold 	100,000*
Mexico (AMPROFON)[183] 	Gold 	30,000*
New Zealand (RMNZ)[72] 	4× Platinum 	120,000‡
Poland (ZPAV)[184] 	3× Platinum 	150,000‡
Portugal (AFP)[185] 	Gold 	10,000‡
Spain (PROMUSICAE)[186] 	Gold 	30,000‡
Sweden (GLF)[187] 	Gold 	20,000‡
Switzerland (IFPI Switzerland)[68] 	Platinum 	30,000^
United Kingdom (BPI)[70] 	3× Platinum 	1,800,000‡
United States (RIAA)[55] 	7× Platinum 	7,000,000‡

* Sales figures based on certification alone.
^ Shipments figures based on certification alone.
‡ Sales+streaming figures based on certification alone.
"I Knew You Were Trouble (Taylor's Version)"
"I Knew You Were Trouble (Taylor's Version)"
Song by Taylor Swift
from the album Red (Taylor's Version)
Released	November 12, 2021
Studio	

    Conway Recording (Los Angeles)
    Kitty Committee (Belfast)

Length	3:39
Label	Republic
Songwriter(s)	

    Taylor Swift Max Martin Shellback

Producer(s)	

    Taylor Swift Shellback Christopher Rowe

Lyric video
"I Knew You Were Trouble (Taylor's Version)" on YouTube

Following the 2019 dispute regarding the ownership of her back catalog, Swift confirmed in November 2020 that she would be re-recording her entire back catalog.[188] Swift previewed the re-recorded version of "I Knew You Were Trouble", subtitled "Taylor's Version", via her Instagram on August 5, 2021.[189] The re-recorded version was produced by Swift, Shellback, and Christopher Rowe.[190] It was engineered and edited at Prime Recording in Nashville, and Swift's vocals were recorded at Conway Recording Studio in Los Angeles and Kitty Committee Studio in Belfast.[191]

"I Knew You Were Trouble (Taylor's Version)" was released as part of her second re-recorded album, Red (Taylor's Version), on November 12, 2021, through Republic Records.[192] Unlike the original track, the title of the re-recorded version is not stylized with a period at the end.[193] Critics complimented the sharper reworked instrumentation for better conveying the emotion.[194][195]

After Red (Taylor's Version) was released, "I Knew You Were Trouble (Taylor's Version)" entered the top 30 of charts in Australia (21),[196] Canada (29),[197] New Zealand (26),[198] and Singapore (13).[199] It peaked at number 23 on the Billboard Global 200.[200] In the US, the re-recording peaked at number 46 on the Billboard Hot 100 and number 16 on Billboard's Adult Contemporary chart.[145][146]
Personnel

Credits are adapted from the liner notes of Red (Taylor's Version).[191]

    Taylor Swift – lead vocals, background vocals, songwriter, producer
    Christopher Rowe – producer, lead vocals engineer
    Shellback – producer, songwriter
    Max Martin – songwriter
    Max Bernstein – synthesizers
    Matt Billingslea – drums programming
    Bryce Bordone – engineer
    Dan Burns – additional programming, additional engineer
    Derek Garten – engineer, editor
    Şerban Ghenea – mixing
    Amos Heller – bass guitar
    Sam Holland – lead vocals engineer
    Mike Meadows – acoustic guitar, synthesizers
    Paul Sidoti – electric guitar

Charts
Weekly charts
Chart performance for "I Knew You Were Trouble (Taylor's Version)" Chart (2021–2024) 	Peak
position
Australia (ARIA)[196] 	21
Canada (Canadian Hot 100)[197] 	29
Canada AC (Billboard)[201] 	27
Canada CHR/Top 40 (Billboard)[202] 	42
Canada Hot AC (Billboard)[203] 	47
Global 200 (Billboard)[200] 	23
New Zealand (Recorded Music NZ)[198] 	26
Poland (Polish Airplay Top 100)[204] 	58
Portugal (AFP)[205] 	85
Singapore (RIAS)[199] 	13
South Africa (RISA)[206] 	92
Sweden Heatseeker (Sverigetopplistan)[207] 	13
UK Audio Streaming (OCC)[208] 	42
US Billboard Hot 100[145] 	46
US Adult Contemporary (Billboard)[146] 	16
US Adult Pop Airplay (Billboard)[209] 	39
	
Year-end charts
2022 year-end chart performance for "I Knew You Were Trouble (Taylor's Version)" Chart (2022) 	Position
US Adult Contemporary (Billboard)[210] 	37

Certifications
Certifications for "I Knew You Were Trouble (Taylor's Version)" Region 	Certification 	Certified units/sales
Australia (ARIA)[71] 	Platinum 	70,000‡
Brazil (Pro-Música Brasil)[179] 	Gold 	20,000‡
New Zealand (RMNZ)[211] 	Gold 	15,000‡
Poland (ZPAV)[212] 	Gold 	25,000‡
United Kingdom (BPI)[213] 	Silver 	200,000‡

‡ Sales+streaming figures based on certification alone.
See also

    List of highest-certified singles in Australia
    List of number-one digital songs of 2012 (U.S.)
    List of number-one digital songs of 2013 (U.S.)
    List of Mainstream Top 40 number-one hits of 2013 (U.S.)
    List of Adult Top 40 number-one singles of 2013

Footnotes

Stylized as "I Knew You Were Trouble." (with a period)[1]

    "We Are Never Ever Getting Back Together" sold 623,000 digital copies in the first week of release.[47]

References

"Red (Deluxe Edition) (2012)". 7digital. May 17, 2019. Archived from the original on July 28, 2021. Retrieved July 27, 2021.
Mansfield, Brian (October 17, 2012). "Taylor Swift Sees Red All Over". USA Today. Archived from the original on December 21, 2012.
Bernstein, Jonathan (November 18, 2020). "500 Greatest Albums: Taylor Swift Looks Back on Her 'Only True Breakup Album' Red". Rolling Stone. Archived from the original on December 4, 2020. Retrieved December 25, 2020.
Rogers, Alex (March 7, 2014). "Q&A: Why Taylor Swift Thinks Nashville Is the Best Place on Earth". Time. Archived from the original on May 24, 2022. Retrieved November 18, 2021.
Gallo, Phil (October 19, 2012). "Taylor Swift Q&A: The Risks of Red and The Joys of Being 22". Billboard. Archived from the original on February 24, 2013.
Dickey, Jack (November 13, 2014). "The Power of Taylor Swift". Time. Archived from the original on August 19, 2020. Retrieved August 8, 2020.
Griffiths, George (June 21, 2021). "The Biggest Hits And Chart Legacy of Taylor Swift's Red Ahead of Its Rerelease". Official Charts Company. Archived from the original on June 21, 2021. Retrieved September 22, 2021.
Shriver, Jerry (October 21, 2012). "Taylor Swift Glows on Hot Red". USA Today. Archived from the original on February 9, 2013. Retrieved September 22, 2021.
Zaleski 2024, p. 82.
Macsai, Dan (October 19, 2012). "Taylor Swift on Going Pop, Ignoring the Gossip and the Best (Worst) Nickname She's Ever Had". Time. Archived from the original on August 14, 2014. Retrieved December 10, 2012.
Light, Alan (December 5, 2014). "Billboard Woman of the Year Taylor Swift on Writing Her Own Rules, Not Becoming a Cliche and the Hurdle of Going Pop". Billboard. Archived from the original on December 26, 2014. Retrieved February 27, 2019.
Swift, Taylor (2012). Red (CD booklet). Big Machine Records. p. 4. 0602537173051.
Maloy, Sarah (October 9, 2012). "Taylor Swift Debuts 'I Knew You Were Trouble' Song: Listen". Billboard. Archived from the original on January 31, 2013. Retrieved December 10, 2012.
Sloan 2021, p. 16.
Hogan, Marc (October 9, 2012). "Hear Taylor Swift's Dubstep-Tinged 'I Knew You Were Trouble'". Spin. Archived from the original on October 5, 2015. Retrieved October 10, 2012.
Roberts, Randall (October 22, 2012). "Album Review: Taylor Swift's Red Burns with Confidence". Los Angeles Times. Archived from the original on November 12, 2012. Retrieved July 21, 2020.
Keefe, Jonathan (October 22, 2012). "Album Review: Red". Slant Magazine. Archived from the original on March 25, 2016. Retrieved April 17, 2015.
Nelson, Brad (August 19, 2019). "Taylor Swift: Red Album Review". Pitchfork. Archived from the original on August 20, 2019. Retrieved July 21, 2020.
Dolan, Jon (October 18, 2012). "Red". Rolling Stone. Archived from the original on January 15, 2018. Retrieved July 24, 2022.
Spencer 2013, p. 122.
Wilman, Chris (October 23, 2012). "Taylor Swift's Red: Track-By-Track". The Hollywood Reporter. Archived from the original on July 30, 2017. Retrieved December 10, 2012.
Perone 2017, p. 45.
Caramanica, Jon (October 24, 2012). "No More Kid Stuff for Taylor Swift". The New York Times. Archived from the original on November 4, 2012. Retrieved November 5, 2012.
Roberts, Randall (October 9, 2012). "First Take: Taylor Swift Accents New Single with Hint of Dubstep". Los Angeles Times. Archived from the original on December 29, 2012. Retrieved December 11, 2012.
Petridis, Alexis (April 29, 2016). "Taylor Swift's Singles – Ranked". The Guardian. Archived from the original on April 27, 2019. Retrieved August 2, 2021.
Reed, James (October 22, 2012). "With Her New Album Red, Taylor Swift Grows Up". The Boston Globe. Archived from the original on June 30, 2017. Retrieved December 11, 2012.
Dobbins, Amanda (October 9, 2012). "Taylor Swift's Version of Dubstep Is a Little Different Than Regular Dubstep". Vulture. Archived from the original on April 21, 2017. Retrieved December 11, 2012.
Stewart, Allison (October 22, 2012). "Taylor Swift's Red Is Another Winner, But She Needs to Start Acting Her Age". The Washington Post. Archived from the original on April 12, 2013. Retrieved December 10, 2012.
"40 Best Songs of 2012". Spin. December 10, 2012. Archived from the original on July 30, 2015. Retrieved July 30, 2015.
"Pazz & Jop: 2012 Singles". The Village Voice. Archived from the original on January 18, 2013. Retrieved January 18, 2013.
Mylrea, Hannah (September 8, 2020). "Every Taylor Swift Song Ranked In Order of Greatness". NME. Archived from the original on September 17, 2020. Retrieved November 26, 2020.
Snapes, Laura (November 12, 2021). "Taylor Swift: Red (Taylor's Version) Review – Getting Back Together with A Classic". The Guardian. Archived from the original on November 12, 2021. Retrieved November 12, 2021.
Bernstein, Alyssa (September 21, 2012). "Taylor Swift Launches Red Album Release With 4-Week Song Preview Countdown". ABC News. Archived from the original on November 23, 2012. Retrieved February 18, 2013.
Vena, Jocelyn (October 8, 2012). "Taylor Swift Falls for a Bad Boy on 'I Knew You Were Trouble'". MTV News. Archived from the original on February 14, 2021. Retrieved December 5, 2012.
Garibaldi, Christina (December 14, 2012). "Taylor Swift Explains Falling for the 'Dangerous' Type... And Why You Should Too". MTV News. Archived from the original on December 8, 2013. Retrieved December 14, 2012.
"I Knew You Were Trouble. – Single by Taylor Swift". Apple Music. Archived from the original on October 11, 2012. Retrieved October 11, 2012.
"Available for Airplay". FMQB. Archived from the original on January 16, 2013. Retrieved December 5, 2012.
Trust, Gary (January 3, 2013). "Bruno Mars 'Locked' at No. 1 on Hot 100, Taylor Swift Closing In". Billboard. Archived from the original on March 8, 2021. Retrieved July 27, 2021.
Aswad, Jem (August 22, 2014). "Taylor Swift & Country: Splitsville!". Billboard. Archived from the original on August 25, 2021. Retrieved August 22, 2014.
"'I Knew You Were Trouble' Single CD". taylorswift.com. Archived from the original on March 24, 2013. Retrieved December 13, 2012.
"'I Knew You Were Trouble' Single Package". taylorswift.com. Archived from the original on January 3, 2013. Retrieved December 13, 2012.
"Singles Release Diary". Digital Spy. Archived from the original on December 3, 2012. Retrieved December 9, 2012.
"Taylor Swift – I Knew You Were Trouble (Universal)". radioairplay.fm. Archived from the original on April 8, 2014. Retrieved April 7, 2014.
Knopper, Steve (November 8, 2014). "Taylor Swift Pulled Music From Spotify for 'Superfan Who Wants to Invest,' Says Rep". Rolling Stone. Archived from the original on April 21, 2015. Retrieved April 21, 2015.
Hern, Alex; Cresci, Elena (December 7, 2015). "Taylor Swift Reappears on Spotify, But Her Music Is Credited to Lostprophets". The Guardian. Archived from the original on December 23, 2015. Retrieved December 25, 2015.
"Taylor Swift Returns to Spotify on the Day Katy Perry's Album Comes Out". BBC News. June 9, 2017. Archived from the original on June 9, 2017. Retrieved June 9, 2017.
Grein, Paul (October 17, 2012). "Week Ending Oct. 14, 2012. Songs: Taylor Swift's Digital Record". Yahoo!. Archived from the original on November 8, 2014. Retrieved October 14, 2012.
Caulfield, Keith (January 3, 2013). "Taylor Swift Leads Record Breaking Digital Sales Week". Billboard. Archived from the original on April 8, 2017. Retrieved August 3, 2021.
Trust, Gary (December 27, 2012). "Bruno Mars Marks a Chart First With Hot 100 Leader 'Heaven'". Billboard. Archived from the original on June 5, 2021. Retrieved August 3, 2021.
Trust, Gary (September 29, 2014). "Chart Highlights: Taylor Swift Tops Adult Pop Songs, Sam Smith Rules Adult R&B". Billboard. Archived from the original on November 10, 2020. Retrieved August 3, 2021.
Jessen, Wade (April 4, 2013). "Darius Rucker Rolls 'Wagon Wheel' to No. 1 On Hot Country Songs". Billboard. Archived from the original on May 28, 2013. Retrieved June 8, 2013.
Trust, Gary (March 4, 2013). "Chart Highlights: Demi Lovato 'Attack's Pop Songs". Billboard. Archived from the original on October 2, 2020. Retrieved March 5, 2013.
Trust, Gary (March 16, 2015). "Chart Highlights: Taylor Swift's 'Style' Fashionably Flies to No. 1 on Pop Songs". Billboard. Archived from the original on October 17, 2019. Retrieved August 3, 2021.
Trust, Gary (July 14, 2019). "Ask Billboard: Taylor Swift's Career Sales & Streaming Totals, From 'Tim McGraw' to 'You Need to Calm Down'". Billboard. Archived from the original on July 15, 2019. Retrieved July 14, 2019.
"American single certifications – Taylor Swift – I Knew You Were Trouble". Recording Industry Association of America. Retrieved March 13, 2015.
"Canadian single certifications – Taylor Swift – I Knew You Were Trouble". Music Canada. Retrieved March 13, 2015.
"ČNS IFPI" (in Czech). Hitparáda – Radio Top 100 Oficiální. IFPI Czech Republic. Note: Select 11. týden 2013 in the date selector. Retrieved November 24, 2013.
"Taylor Swift – I Knew You Were Trouble". Tracklisten. Retrieved November 24, 2013.
"The Irish Charts – Search Results – I Knew You Were Trouble". Irish Singles Chart. Retrieved November 24, 2013.
"Taylor Swift – I Knew You Were Trouble" (in German). Ö3 Austria Top 40. Retrieved November 24, 2013.
"Russia Airplay Chart for 2013-06-03." TopHit. Retrieved November 24, 2013.
Taylor Swift — I Knew You Were Trouble. TopHit. Retrieved December 7, 2020.
"Taylor Swift – I Knew You Were Trouble". Swiss Singles Chart. Retrieved November 24, 2013.
"Taylor Swift – I Knew You Were Trouble" (in German). GfK Entertainment charts. Retrieved November 24, 2013.
"Taylor Swift – I Knew You Were Trouble" (in Dutch). Ultratop 50. Retrieved December 31, 2012.
"Suomen virallinen lista – I Knew You Were Trouble" (in Finnish). Musiikkituottajat. Archived from the original on October 8, 2021. Retrieved October 7, 2021.
"Gold-/Platin-Datenbank (Taylor Swift; 'I Knew You Were Trouble')" (in German). Bundesverband Musikindustrie. Retrieved April 28, 2018.
"The Official Swiss Charts and Music Community: Awards ('I Knew You Were Trouble')". IFPI Switzerland. Hung Medien. Retrieved March 13, 2015.
"Official Singles Chart Top 100". Official Charts Company. Retrieved November 24, 2013.
"British single certifications – Taylor Swift – I Knew You Were Trouble". British Phonographic Industry. Retrieved July 20, 2024.
"ARIA Charts – Accreditations – 2024 Singles" (PDF). Australian Recording Industry Association. Retrieved February 14, 2024.
"New Zealand single certifications – Taylor Swift – I Knew You Were Trouble". Radioscope. Retrieved December 19, 2024. Type I Knew You Were Trouble in the "Search:" field.
Martins, Chris (December 13, 2012). "Taylor Swift's 'I Knew You Were Trouble' Video Is Lana Del Rey's 'Ride' All Over Again". Spin. Archived from the original on October 5, 2015. Retrieved December 20, 2012.
Schillaci, Sophie (December 10, 2012). "Taylor Swift I Knew MV Premiere". The Hollywood Reporter. Archived from the original on August 9, 2020. Retrieved December 10, 2012.
Montgomery, James (December 10, 2012). "Taylor Swift to Premiere 'I Knew You Were Trouble' Video on MTV!". MTV News. Archived from the original on December 7, 2013. Retrieved December 11, 2012.
Newman, Melinda (December 13, 2012). "Watch: Taylor Swift Gets Led Astray in New Video for 'I Knew You Were Trouble'". Uproxx. Archived from the original on August 27, 2021. Retrieved December 13, 2012.
Geller, Wendy (December 14, 2012). "Taylor Swift Shows Gritty, Sexy, Very Non-Country Style in New Video". Yahoo!. Archived from the original on August 27, 2021. Retrieved December 14, 2012.
"Taylor Swift Debuts Edgy New Look in I Knew You Were Trouble Video". Marie Claire. December 14, 2012. Archived from the original on November 25, 2020. Retrieved August 3, 2021.
"Taylor Goes Punk in 'I Knew You Were Trouble'". Rolling Stone. December 14, 2012. Archived from the original on December 16, 2012. Retrieved December 14, 2012.
Brodsky, Rachel (December 13, 2012). "Video Premiere: Taylor Swift, 'I Knew You Were Trouble.'". MTV News. Archived from the original on August 3, 2021. Retrieved December 13, 2012.
Dobbins, Amanda (December 13, 2012). "Watch T-Swift's 'I Knew You Were Trouble' Video". Vulture. Archived from the original on March 14, 2016. Retrieved December 20, 2012.
Brennan, Matt (April 2, 2013). "Now and Then: Why We Love GIFs, from Taylor Swift to Goats". IndieWire. Archived from the original on September 21, 2021. Retrieved August 30, 2021.
Bruner, Raisa (November 12, 2018). "The New 'Grinch' Movie Is Hiding a Taylor Swift Meme in Plain Sight". Time. Archived from the original on November 12, 2018. Retrieved November 12, 2018.
"Stevie Nicks and Top Songwriters Honored at 62nd Annual BMI Pop Awards". Broadcast Music, Inc. May 14, 2014. Archived from the original on April 3, 2016. Retrieved May 11, 2016.
"Most Performed Songs". American Society of Composers, Authors and Publishers. Archived from the original on January 27, 2016. Retrieved April 15, 2015.
"Gomez, Bieber Win Top Disney Gongs". Irish Independent. April 29, 2013. Archived from the original on April 29, 2013. Retrieved April 29, 2013.
"VMAs: The 2013 Winner's List". Entertainment Weekly. August 25, 2013. Archived from the original on August 27, 2013. Retrieved August 26, 2013.
Spangler, Todd (October 21, 2013). "YouTube Music Awards Nominees Announced". Variety. Archived from the original on July 1, 2015. Retrieved July 7, 2015.
"Nickelodeon Unveils 2013 Kids' Choice Award Nominees". Foxtel. February 14, 2013. Archived from the original on March 3, 2013. Retrieved January 25, 2015.
"2013 Teen Choice Awards: The Winners List". MTV News. August 11, 2013. Archived from the original on March 3, 2016. Retrieved June 21, 2015.
Couch, Aaron; Washington, Arlene (March 29, 2014). "Kids' Choice Awards: The 2014 Winners Announced". The Hollywood Reporter. Archived from the original on September 7, 2021. Retrieved June 23, 2015.
"Taylor Swift Strikes AMA Stage with Chaotic Performance". MTV News. November 18, 2012. Archived from the original on November 1, 2020. Retrieved August 3, 2021.
Pacella, Megan (November 27, 2012). "Taylor Swift Performs Red Songs on 'The Today Show Australia'". Taste of Country. Archived from the original on November 29, 2012. Retrieved November 27, 2012.
McCabe, Kathy; Christie, Joel (November 29, 2012). "Gotye Takes Out ARIA Album of the Year, Male Artist of the Year". News.com.au. Archived from the original on August 4, 2021. Retrieved November 29, 2012.
Brown, August (December 2, 2012). "Review: Taylor Swift, Ne-Yo and Jonas Brothers at KIIS' Jingle Ball". Los Angeles Times. Archived from the original on August 4, 2021. Retrieved December 2, 2012.
Hampp, Andrew (December 8, 2012). "Justin Bieber, Taylor Swift, One Direction and More Light Up NYC at Z100 Jingle Ball". Billboard. Archived from the original on March 9, 2016. Retrieved April 28, 2013.
Vena, Jocelyn (January 1, 2013). "Taylor Swift, Harry Styles Kick Off New Year with a Kiss". MTV News. Archived from the original on November 7, 2017. Retrieved January 1, 2013.
"On y était : Taylor Swift en concert privé NRJ !". NRJ (in French). January 28, 2013. Archived from the original on September 16, 2015. Retrieved January 28, 2013.
Vena, Jocelyn (February 20, 2013). "Taylor Swift Gets Into 'Trouble' During BRIT Awards". MTV News. Archived from the original on November 7, 2017. Retrieved November 3, 2017.
"Taylor Swift : I knew You Were Trouble, son incroyable performance au Graham Norton Show". Melty (in French). February 23, 2013. Archived from the original on April 3, 2013. Retrieved August 3, 2021.
Sheffield, Rob (March 28, 2013). "Taylor Swift's Red Tour: Her Amps Go Up to 22". Rolling Stone. Archived from the original on March 26, 2014. Retrieved October 19, 2017.
Levy, Piet (August 11, 2013). "Concert Review: Taylor Swift's Red Tour Brings Color, Spectacle to Chicago's Soldier Field". Milwaukee Journal Sentinel. Archived from the original on August 5, 2021. Retrieved August 4, 2021.
Ford, Rebecca (August 20, 2013). "Taylor Swift Finds Love in Los Angeles: Concert Review". The Hollywood Reporter. Archived from the original on August 5, 2021. Retrieved August 21, 2013.
DelliCarpini Jr., Gregory (November 14, 2013). "Taylor Swift, Fall Out Boy and More Perform at Victoria's Secret Fashion Show: Watch". Billboard. Archived from the original on December 7, 2016. Retrieved November 3, 2017.
Lipshutz, Jason (September 20, 2014). "Taylor Swift Shakes Off the 'Frenemies' During iHeartRadio Fest Performance: Watch". Billboard. Archived from the original on September 20, 2014. Retrieved December 9, 2019.
Edwards, Gavin (October 25, 2014). "Taylor Swift, Ariana Grande and Gwen Stefani Cover the Hollywood Bowl in Glitter". Rolling Stone. Archived from the original on February 28, 2019. Retrieved December 15, 2020.
Stutz, Colin (December 6, 2014). "Taylor Swift Beats Laryngitis, Sam Smith, Ariana Grande Shine at KIIS FM Jingle Ball". Billboard. Archived from the original on January 2, 2021. Retrieved December 15, 2020.
Sheffield, Rob (July 11, 2015). "Taylor Swift's Epic 1989 Tour: Every Night With Us Is Like a Dream". Rolling Stone. Archived from the original on April 13, 2020. Retrieved December 1, 2020.
Caramanica, Jon (May 21, 2015). "Review: On Taylor Swift's '1989' Tour, the Underdog Emerges as Cool Kid". The New York Times. Archived from the original on November 28, 2019. Retrieved November 28, 2019.
Iasimone, Ashley (May 27, 2018). "All the Surprise Songs Taylor Swift Has Performed On Her Reputation Stadium Tour B-Stage (So Far)". Billboard. Archived from the original on May 27, 2018. Retrieved December 19, 2018.
Willman, Chris (June 2, 2019). "Taylor Swift Goes Full Rainbow for Pride Month at L.A. Wango Tango Show". Variety. Archived from the original on December 7, 2020. Retrieved December 16, 2020.
Brandle, Lars (July 11, 2019). "Taylor Swift Sings 'Shake It Off,' 'Blank Space' & More at Amazon Prime Day Concert: Watch". Billboard. Archived from the original on July 12, 2019. Retrieved July 14, 2019.
Mylrea, Hannah (September 10, 2019). "Taylor Swift's The City of Lover concert: a triumphant yet intimate celebration of her fans and career". NME. Archived from the original on September 16, 2019. Retrieved September 12, 2019.
Gracie, Bianca (November 24, 2019). "Taylor Swift Performs Major Medley Of Hits, Brings Out Surprise Guests For 'Shake It Off' at 2019 AMAs". Billboard. Archived from the original on November 26, 2019. Retrieved November 25, 2019.
Shafer, Ellise (March 18, 2023). "Taylor Swift Eras Tour: The Full Setlist From Opening Night". Variety. Archived from the original on March 18, 2023. Retrieved March 19, 2023.
"We Came As Romans cover Swift's Trouble". Metal Hammer. October 15, 2014.
Paul, Larisha (October 18, 2023). "Sabrina Carpenter Takes on Taylor Swift's 'I Knew You Were Trouble' for Spotify Singles". Rolling Stone. Archived from the original on November 25, 2023. Retrieved September 16, 2024.
"Taylor Swift – I Knew You Were Trouble". ARIA Top 50 Singles. Retrieved December 10, 2012.
"Taylor Swift – I Knew You Were Trouble" (in French). Ultratop 50. Retrieved November 24, 2013.
BPP, ed. (May 2013). "Billboard Brasil Hot 100 Airplay". Billboard Brasil (40): 84–89.
"17.12.2012–23.12.2012" Airplay Top 5. Bulgarian Association of Music Producers. Retrieved December 23, 2012.
"Taylor Swift Chart History (Canadian Hot 100)". Billboard. Retrieved October 18, 2012.
"Taylor Swift Chart History (Canada AC)". Billboard. Retrieved November 24, 2013.
"Taylor Swift Chart History (Canada CHR/Top 40)". Billboard. Retrieved November 24, 2013.
"Taylor Swift Chart History (Canada Hot AC)". Billboard. Retrieved November 24, 2013.
"Croatia ARC TOP 20". HRT. Retrieved December 6, 2013.[permanent dead link]
"Taylor Swift Chart History (Euro Digital Song Sales)". Billboard. Retrieved November 24, 2013.
"Taylor Swift – I Knew You Were Trouble" (in French). Les classement single. Retrieved November 24, 2013.
"Archívum – Slágerlisták – MAHASZ" (in Hungarian). Rádiós Top 40 játszási lista. Magyar Hanglemezkiadók Szövetsége. Retrieved November 24, 2013.
"Media Forest Week 05, 2013". Israeli Airplay Chart. Media Forest. Retrieved November 24, 2013.
"Japan Billboard Hot 100". Billboard Japan (in Japanese). January 21, 2013. Archived from the original on March 3, 2016. Retrieved August 25, 2015.
"Japan Adult Contemporary Airplay Chart". Billboard Japan (in Japanese). Retrieved October 31, 2023.
"Taylor Swift Chart History". The Official Lebanese Top 20. Archived from the original on September 17, 2016. Retrieved January 6, 2018.
"Taylor Swift Chart History (Luxembourg Digital Song Sales)". Billboard. Retrieved November 24, 2013. [dead link]
"Top 20 Inglés Del 4 al 10 de Marzo, 2013". Monitor Latino. March 10, 2013. Retrieved May 2, 2018.
"Nederlandse Top 40 – Taylor Swift" (in Dutch). Dutch Top 40. Retrieved November 24, 2013.
"Taylor Swift – I Knew You Were Trouble" (in Dutch). Single Top 100. Retrieved November 24, 2013.
"Taylor Swift – I Knew You Were Trouble". Top 40 Singles. Retrieved November 24, 2013.
"Official Scottish Singles Sales Chart Top 100". Official Charts Company. Retrieved November 24, 2013.
"ČNS IFPI" (in Slovak). Hitparáda – Radio Top 100 Oficiálna. IFPI Czech Republic. Note: insert 201315 into search. Retrieved November 24, 2013.
"Taylor Swift – I Knew You Were Trouble" Canciones Top 50. Retrieved November 24, 2013.
"Taylor Swift – I Knew You Were Trouble". Singles Top 100. Retrieved November 24, 2013.
"Number One Top 20 | Klip Izle" (in Turkish). Number One Top 20. May 25, 2013. Archived from the original on June 1, 2013. Retrieved October 7, 2022.
"Ukraine Airplay Chart for 2013-05-27." TopHit. Retrieved November 24, 2013.
"Taylor Swift Chart History (Hot 100)". Billboard. Retrieved November 24, 2013.
"Taylor Swift Chart History (Adult Contemporary)". Billboard. Retrieved November 24, 2013.
"Taylor Swift Chart History (Adult Pop Songs)". Billboard. Retrieved November 24, 2013.
"Taylor Swift Chart History (Country Airplay)". Billboard. Retrieved November 24, 2013.
"Taylor Swift Chart History (Dance Mix/Show Airplay)". Billboard. Retrieved November 24, 2013.
"Taylor Swift Chart History (Pop Songs)". Billboard. Retrieved November 24, 2013.
"Taylor Swift Chart History (Rhythmic)". Billboard. Retrieved November 24, 2013.
"ARIA End of Year Singles Chart 2012". Australian Recording Industry Association. Archived from the original on February 24, 2020. Retrieved June 18, 2021.
"Top Selling Singles of 2012". Recorded Music NZ. Archived from the original on August 12, 2021. Retrieved August 11, 2021.
"End of Year 2012" (PDF). UKChartsPlus. Archived (PDF) from the original on March 31, 2016. Retrieved April 29, 2020.
"ARIA End of Year Singles Chart 2013". Australian Recording Industry Association. Archived from the original on September 4, 2019. Retrieved March 15, 2020.
"Jahreshitparade Singles 2013" (in German). Hung Medien. Archived from the original on February 24, 2020. Retrieved January 5, 2020.
"Jaaroverzichten 2013" (in Dutch). Ultratop. Archived from the original on April 17, 2014. Retrieved October 23, 2019.
"Rapports Annuels 2013" (in French). Ultratop. Archived from the original on June 8, 2014. Retrieved November 3, 2019.
"Best of 2013: Canadian Hot 100". Billboard. January 2, 2013. Archived from the original on December 14, 2013. Retrieved December 13, 2013.
"Track Top-50 2013" (in Danish). hitlisterne.dk. Archived from the original on February 4, 2014. Retrieved March 6, 2014.
"Top de l'année Top Singles 2013" (in French). Syndicat National de l'Édition Phonographique. Archived from the original on September 22, 2020. Retrieved November 21, 2020.
"Top 100 Single – Jahrescharts 2013" (in German). GfK Entertainment charts. Archived from the original on May 9, 2015. Retrieved October 6, 2021.
"Mahasz Rádiós Top 100 – radios 2013" (in Hungarian). Association of Hungarian Record Companies. Archived from the original on October 7, 2020. Retrieved January 22, 2014.
"IRMA - best of 2013". IRMA. Archived from the original on September 6, 2018. Retrieved May 26, 2022.
"Top 100-Jaaroverzicht van 2013" (in Dutch). Dutch Top 40. Archived from the original on March 29, 2019. Retrieved October 23, 2019.
"Top Selling Singles of 2013". Recorded Music NZ. Archived from the original on October 29, 2014. Retrieved January 1, 2015.
"Russian Top Year-End Radio Hits (2013)". TopHit. Archived from the original on July 3, 2019. Retrieved August 10, 2019.
"Schweizer Jahreshitparade 2013" (in German). Hung Medien. Archived from the original on August 13, 2014. Retrieved October 23, 2019.
"Ukrainian Top Year-End Radio Hits (2013)". TopHit. Archived from the original on July 3, 2019. Retrieved August 10, 2019.
Lane, Daniel (January 1, 2014). "The Official Top 40 Biggest Selling Singles Of 2013". Official Charts Company. Archived from the original on February 23, 2015. Retrieved January 1, 2014.
"Best of 2013 – Hot 100 Songs". Billboard. Archived from the original on November 27, 2015. Retrieved December 13, 2013.
"Adult Contemporary Songs – Year-End 2013". Billboard. Archived from the original on October 28, 2020. Retrieved August 31, 2019.
"Adult Pop Songs – Year-End 2013". Billboard. Archived from the original on February 24, 2020. Retrieved August 31, 2019.
"Pop Songs – Year-End 2013". Billboard. Archived from the original on February 24, 2020. Retrieved August 31, 2019.
"2013 Year End Charts – Top Billboard Radio Songs". Billboard. Archived from the original on September 21, 2017. Retrieved April 18, 2014.
"Greatest of All Time Pop Songs". Billboard. Archived from the original on June 13, 2018. Retrieved August 1, 2018.
"Austrian single certifications – Taylor Swift – I Knew You Were Trouble" (in German). IFPI Austria. Retrieved May 29, 2024.
"Goud en Platina 2013" (in Dutch). Ultratop. Archived from the original on March 8, 2021. Retrieved January 10, 2021.
"Brazilian single certifications – Taylor Swift – I Knew You Were Trouble" (in Portuguese). Pro-Música Brasil. Retrieved July 23, 2024.
"Danish single certifications – Taylor Swift – I Knew You Were Trouble". IFPI Danmark. Retrieved October 5, 2021.
"Italian single certifications – Taylor Swift – I Knew You Were Trouble" (in Italian). Federazione Industria Musicale Italiana. Retrieved December 16, 2019.
"Japanese digital single certifications – Taylor Swift – I Knew You Were Trouble" (in Japanese). Recording Industry Association of Japan. Retrieved January 27, 2016. Select 2015年1月 on the drop-down menu
"Certificaciones" (in Spanish). Asociación Mexicana de Productores de Fonogramas y Videogramas. Retrieved March 13, 2015. Type Taylor Swift in the box under the ARTISTA column heading and I Knew You Were Trouble in the box under the TÍTULO column heading.
"Przyznane w 2021 roku" (in Polish). Polish Society of the Phonographic Industry. Archived from the original on September 22, 2021. Retrieved August 11, 2021.
"Portuguese single certifications" (PDF) (in Portuguese). Associação Fonográfica Portuguesa. Retrieved May 23, 2022.
"Spanish single certifications – Taylor Swift – I Knew You Were Trouble". El portal de Música. Productores de Música de España. Retrieved April 1, 2024.
"Taylor Swift" (in Swedish). Sverigetopplistan. Archived from the original on August 3, 2021. Retrieved August 2, 2021.
Willman, Chris (November 16, 2020). "Taylor Swift Confirms Sale of Her Masters, Says She Is Already Re-Recording Her Catalog". Variety. Archived from the original on November 17, 2020. Retrieved November 18, 2020.
Willman, Chris (August 5, 2021). "Taylor Swift Teases Phoebe Bridgers, Chris Stapleton Collaborations for Red Album in Word Puzzle". Variety. Archived from the original on August 9, 2021. Retrieved September 23, 2021.
"Credits / Red (Taylor's Version) / Taylor Swift". Tidal. November 12, 2021. Archived from the original on November 13, 2021. Retrieved November 18, 2021.
Swift, Taylor (2021). Red (Taylor's Version) (vinyl liner notes). Republic Records.
Al-Heeti, Abrar (November 11, 2021). "Red (Taylor's Version): Release Date, Tracklist, Why Taylor Swift Is Rerecording Her Albums". CNET. Archived from the original on November 20, 2021. Retrieved November 13, 2021.
"Red (Taylor's Version) (+ A Message from Taylor) by Taylor Swift". Apple Music. Archived from the original on October 28, 2021. Retrieved November 16, 2021.
Brown, Helen (November 12, 2021). "Taylor Swift's Red Is A Better, Brighter Version of A Terrific Pop Album". The Independent. Archived from the original on November 12, 2021. Retrieved November 12, 2021.
Keefe, Jonathan (November 17, 2021). "Review: Taylor Swift's Red Redux Flaunts the Singer's Refined Pop Instincts". Slant Magazine. Archived from the original on November 17, 2021. Retrieved November 22, 2021.
"ARIA Top 50 Singles Chart". Australian Recording Industry Association. November 22, 2021. Archived from the original on November 19, 2021. Retrieved November 19, 2021.
"Taylor Swift Chart History (Canadian Hot 100)". Billboard. Retrieved November 23, 2021.
"NZ Top 40 Singles Chart". Recorded Music NZ. November 22, 2021. Archived from the original on November 19, 2021. Retrieved November 20, 2021.
"RIAS Top Charts Week 46 (12 – 18 Nov 2021)". Recording Industry Association Singapore. November 23, 2021. Archived from the original on November 23, 2021. Retrieved November 23, 2021.
"Taylor Swift Chart History (Global 200)". Billboard. Retrieved November 23, 2021.
"Taylor Swift Chart History (Canada AC)". Billboard. Retrieved April 5, 2022.
"Taylor Swift Chart History (Canada CHR/Top 40)". Billboard. Retrieved March 29, 2022.
"Taylor Swift Chart History (Canada Hot AC)". Billboard. Retrieved January 24, 2022.
"OLiS – oficjalna lista airplay" (Select week 13.04.2024–19.04.2024.) (in Polish). OLiS. Retrieved April 22, 2024.
"Taylor Swift – I Knew You Were Trouble". AFP Top 100 Singles. Retrieved February 9, 2022.
"Local & International Streaming Chart Top 100: Week 46". Recording Industry of South Africa. Archived from the original on November 25, 2021. Retrieved November 26, 2021.
"Veckolista Heatseeker, vecka 21". Sverigetopplistan. Retrieved May 24, 2024.
"Official Audio Streaming Chart Top 100". Official Charts Company. Retrieved November 19, 2021.
"Taylor Swift Chart History (Adult Pop Songs)". Billboard. Retrieved June 28, 2022.
"Adult Contemporary Songs – Year-End 2022". Billboard. Archived from the original on December 2, 2022. Retrieved December 2, 2022.
"New Zealand single certifications – Taylor Swift – I Knew You Were Trouble (Taylor's Version)". Radioscope. Retrieved December 19, 2024. Type I Knew You Were Trouble (Taylor's Version) in the "Search:" field.
"OLiS - oficjalna lista wyróżnień" (in Polish). Polish Society of the Phonographic Industry. Retrieved December 18, 2024. Click "TYTUŁ" and enter I Knew You Were Trouble (Taylor's Version) in the search box.

    "British single certifications – Taylor Swift – I Knew You Were Trouble (Taylor's Version)". British Phonographic Industry. Retrieved November 24, 2023.

Source

    Perone, James E. (2017). "Red". The Words and Music of Taylor Swift. The Praeger Singer-Songwriter Collection. ABC-Clio. pp. 43–54. ISBN 978-1-44-085294-7.
    Sloan, Nate (2021). "Taylor Swift and the Work of Songwriting". Contemporary Music Review. 40 (1): 11–26. doi:10.1080/07494467.2021.1945226.
    Spencer, Liv (2013). "Swift Notes". Taylor Swift. ECW Press. pp. 119–130. ISBN 978-1-77041-151-7.
    Zaleski, Annie (2024). "The Red Era". Taylor Swift: The Stories Behind the Songs. Thunder Bay Press. pp. 76–105. ISBN 978-1-6672-0845-9.

    vte

Taylor Swift songs

    Singles discography

Taylor Swift	

    "Tim McGraw" "Picture to Burn" "Teardrops on My Guitar" "A Place in This World" "Should've Said No" "Our Song"

Fearless	

    "Fearless" "Fifteen" "Love Story" "Hey Stephen" "White Horse" "You Belong with Me" "Breathe" "You're Not Sorry" "The Way I Loved You" "Forever & Always" "The Best Day" "Change"

Taylor's Version	

    "You All Over Me" "Mr. Perfectly Fine" "That's When"

Speak Now	

    "Mine" "Sparks Fly" "Back to December" "Speak Now" "Dear John" "Mean" "The Story of Us" "Never Grow Up" "Enchanted" "Better than Revenge" "Innocent" "Haunted" "Long Live" "Ours"

Taylor's Version	

    "Electric Touch" "When Emma Falls in Love" "I Can See You" "Castles Crumbling"

Red	

    "State of Grace" "Red" "Treacherous" "I Knew You Were Trouble" "All Too Well" "22" "We Are Never Ever Getting Back Together" "The Last Time" "Holy Ground" "Everything Has Changed" "Begin Again"

Taylor's Version	

    "Nothing New" "Message in a Bottle" "I Bet You Think About Me" "Forever Winter"

1989	

    "Welcome to New York" "Blank Space" "Style" "Out of the Woods" "All You Had to Do Was Stay" "Shake It Off" "I Wish You Would" "Bad Blood" "Wildest Dreams" "How You Get the Girl" "This Love" "I Know Places" "Clean" "You Are in Love" "New Romantics"

Taylor's Version	

    "'Slut!'" "Say Don't Go" "Now That We Don't Talk" "Suburban Legends" "Is It Over Now?"

Reputation	

    "...Ready for It?" "End Game" "I Did Something Bad" "Don't Blame Me" "Delicate" "Look What You Made Me Do" "So It Goes..." "Gorgeous" "Getaway Car" "Dress" "Call It What You Want" "New Year's Day"

Lover	

    "I Forgot That You Existed" "Cruel Summer" "Lover" "The Man" "The Archer" "Miss Americana & the Heartbreak Prince" "Paper Rings" "Cornelia Street" "Death by a Thousand Cuts" "London Boy" "Soon You'll Get Better" "False God" "You Need to Calm Down" "Me!"

Folklore	

    "The 1" "Cardigan" "The Last Great American Dynasty" "Exile" "My Tears Ricochet" "Mirrorball" "Seven" "August" "This Is Me Trying" "Illicit Affairs" "Invisible String" "Mad Woman" "Epiphany" "Betty" "Peace" "Hoax" "The Lakes"

Evermore	

    "Willow" "Champagne Problems" "'Tis the Damn Season" "Tolerate It" "No Body, No Crime" "Happiness" "Dorothea" "Coney Island" "Long Story Short" "Marjorie"

Midnights	

    "Lavender Haze" "Maroon" "Anti-Hero" "Snow on the Beach" "You're on Your Own, Kid" "Midnight Rain" "Question...?" "Vigilante Shit" "Bejeweled" "Labyrinth" "Karma" "Sweet Nothing" "Mastermind" "Hits Different" "Bigger Than the Whole Sky" "Would've, Could've, Should've" "You're Losing Me"

The Tortured Poets
Department	

    "Fortnight" "The Tortured Poets Department "My Boy Only Breaks His Favorite Toys" "Down Bad" "So Long, London" "But Daddy I Love Him" "Fresh Out the Slammer" "Florida!!!" "Guilty as Sin?" "Who's Afraid of Little Old Me?" "Loml" "I Can Do It with a Broken Heart" "The Smallest Man Who Ever Lived" "The Alchemy" "Clara Bow" "The Black Dog" "So High School" "Thank You Aimee"

Soundtrack songs	

    "Crazier" "Today Was a Fairytale" "Safe & Sound" "Eyes Open" "Sweeter than Fiction" "I Don't Wanna Live Forever" "Beautiful Ghosts" "Only the Young" "Carolina"

Featured songs	

    "Two Is Better Than One" "Half of My Heart" "Both of Us" "Babe" "Gasoline" "Renegade" "The Joker and the Queen" "The Alcott" "Us"

Other songs	

    "Best Days of Your Life" "Ronan" "Highway Don't Care" "Better Man" "Christmas Tree Farm" "All of the Girls You Loved Before"

    Category

    vte

YouTube Music Awards

    vte

MTV Video Music Award for Best Female Video
Authority control databases Edit this at Wikidata	

    MusicBrainz workMusicBrainz release group

Categories:

    2012 singles2012 songsAmerican pop rock songsTaylor Swift songsBig Machine Records singlesSongs written by Taylor SwiftSongs written by Max MartinSongs written by Shellback (record producer)Song recordings produced by Max MartinSong recordings produced by Shellback (record producer)Song recordings produced by Taylor SwiftSong recordings produced by Chris RoweMusic videos directed by Anthony MandlerMTV Video Music Award for Best Female VideoInternet memes introduced in 2012American dance-pop songsTeen pop songs

    This page was last edited on 19 December 2024, at 22:09 (UTC).
    Text is available under the Creative Commons Attribution-ShareAlike 4.0 License; additional terms may apply. By using this site, you agree to the Terms of Use and Privacy Policy. Wikipedia® is a registered trademark of the Wikimedia Foundation, Inc., a non-profit organization.

    Privacy policy
    About Wikipedia
    Disclaimers
    Contact Wikipedia
    Code of Conduct
    Developers
    Statistics
    Cookie statement
    Mobile view

    Wikimedia Foundation
    Powered by MediaWiki



Wikipedia The Free Encyclopedia

    Donate
    Create account
    Log in

Contents
(Top)
Background and release

Music

Lyrics and interpretations

Critical reviews

Charts

Certifications

2021 re-recordings

    Release
    Lyrics
    Critical reception
    Accolades
    Commercial performance
    Charts
        Weekly charts
        Year-end charts
    Certifications
    Release history

Live performances

Impact and legacy

    Listicles
    Recognition
    The scarf

Personnel

Notes

References

        Sources
    External links

All Too Well

    Article
    Talk

    Read
    Edit
    View history

Tools

Appearance
Text

    Small
    Standard
    Large

Width

    Standard
    Wide

Color (beta)

    Automatic
    Light
    Dark

From Wikipedia, the free encyclopedia
For the film, see All Too Well: The Short Film. For other uses, see All Too Well (disambiguation).
"All Too Well"
Song by Taylor Swift
from the album Red
Written	2011
Released	October 22, 2012
Studio	Pain in the Art (Nashville)
Genre	

    Country rock soft rock folk-rock

Length	5:29
Label	Big Machine
Songwriter(s)	

    Taylor Swift Liz Rose

Producer(s)	

    Taylor Swift Nathan Chapman

Audio
"All Too Well" on YouTube

"All Too Well" is a song by the American singer-songwriter Taylor Swift. Written by Swift and Liz Rose, the song was first produced by Swift and Nathan Chapman for her fourth studio album, Red (2012). After a 2019 dispute regarding the ownership of Swift's masters, she re-recorded the song as "All Too Well (Taylor's Version)" and released an unabridged "10 Minute Version" as part of the re-recorded album Red (Taylor's Version) in November 2021.[note 1]

The lyrics of "All Too Well" narrate a failed romantic relationship, recalling the intimate memories and exploring the painful aftermath. The detail of a scarf that the narrator left at the house of her ex-lover's sister generated widespread interpretations and became a popular culture phenomenon. The 2012 version is a slow-burning power ballad combining styles of country rock, soft rock and folk-rock. The "10 Minute Version", produced by Swift and Jack Antonoff, has an atmospheric pop rock production. Swift performed the song at the 2014 Grammy Awards and included it in the set lists for two of her world tours: the Red Tour (2013–2014) and the Eras Tour (2023–2024).

"All Too Well" charted in Canada and the United States in 2012, and the "Taylor's Version" re-recording peaked atop the Billboard Global 200 and became the longest song to top the US Billboard Hot 100; it reached number one on charts in several other countries. Music critics unanimously regard "All Too Well" as Swift's masterpiece and praise its evocative and detail-heavy lyricism. Rolling Stone included it at number 69 in their 2021 revision of the 500 Greatest Songs of All Time. Critics praised the "10 Minute Version" for providing a richer context with its additional verses; it received a Grammy nomination for Song of the Year. It was accompanied by a short film directed by Swift, which won the Grammy Award for Best Music Video.
Background and release
Swift (pictured in 2012) began writing the song during rehearsals for the Speak Now World Tour in February 2011.

Taylor Swift was inspired by her tumultuous feelings after a breakup while conceiving her fourth studio album, Red (2012).[1] "All Too Well" was the first song Swift wrote for Red.[2] During a February 2011 rehearsal of the Speak Now World Tour, she ad-libbed some lyrics to the song while playing a four-chord guitar riff as her touring band spontaneously played backing instruments.[1] Swift told Rolling Stone that this relationship caused "a few roller coasters", and she channeled the tumult into the song.[1] According to the booklets of Swift's 2019 album Lover, the final draft was completed a month later.[3][4]

Swift asked Liz Rose, who had co-written songs on her first albums, to co-write "All Too Well". Rose said that it was an unexpected collaboration after not having worked with Swift for some years;[5] the last time they worked together was on Swift's 2008 album Fearless.[6] When Rose agreed to collaborate, she recalled that Swift had come up with the melody.[7] In an interview with Good Morning America, Swift said that the song was "the hardest to write on the album", saying that it took her some time "to filter through everything [she] wanted to put in the song" without making it lengthy with the help from Rose.[8] Rose said that the song was originally "10, 12 or 15 minutes long" or "probably a 20-minute song" before trimming.[5][9] The final album version is 5 minutes and 28 seconds long, the longest song on Red.[10] It was produced by Swift and Nathan Chapman.[11]

Red was released on October 22, 2012, by Big Machine Records.[12] After Red was released, "All Too Well" debuted at number 80 on the Billboard Hot 100 chart dated November 10, 2012;[13] it also charted at number 17 on Hot Country Songs[14] and number 58 on Country Airplay.[15] The Recording Industry Association of America (RIAA) in 2018 certified the track gold for crossing 500,000 units based on sales and streams.[16] The song also peaked at number 59 on the Canadian Hot 100,[17] and it was certified platinum in Australia,[18] gold in Brazil[19] and Portugal,[20] and silver in the United Kingdom.[21]
Music
"All Too Well"
Duration: 32 seconds.0:32
Chronicling a fallen relationship, "All Too Well" builds up with overdubbed guitars, drums, and strings. Its sound blends styles of country, folk, and rock.
Problems playing this file? See media help.

"All Too Well" is a power ballad[22][23] that is set over a 4/4 beat.[6] The track gradually builds up to a big, eruptive crescendo in each refrain.[6][24][25] Chapman is the sole musician on the track; he played acoustic guitar, electric guitar, bass, and drums, and provided background vocals. The instruments are overdubbed and multitracked.[26] Jody Rosen, in an article for New York, wrote that the production "rises, like a slow-cresting wave".[27]

Bernard Perusse of the Edmonton Journal wrote that the song displays influences of U2,[28] while Rosen and Spin's Michael Robbins thought that its driving bassline was reminiscent of that from U2's "With or Without You" (1987).[27][29] Slant Magazine's Jonathan Keefe wrote that the track progresses from "coffeehouse folk to arena rock".[24] Bruce Warren, assistant general manager of programming at WXPN, commented that the track has a soft rock production.[23] Billboard and Craig S. Semon of the Telegram & Gazette described the genre as country,[30][31] while Newsday said that the song has an "alt-country ache",[32] and Jon Dolan of Rolling Stone thought that "All Too Well" exemplified the "diaristic post-country rock" sound of Red.[33]
Lyrics and interpretations

The song is a recount of a fallen relationship[6][30] that happened in the fall.[34][35][36] The verses explore the memories in a loosely chronological order,[37] using expository details.[38] The song begins with details of a trip of two ex-lovers to Upstate New York ("We're singin' in the car, getting lost upstate")[25] such as imagery of "autumn leaves falling down like pieces into place".[35] When they visit the house of the love interest's sister during the Thanksgiving weekend, the narrator leaves her scarf at that place.[25][39][40] The narrator then recalls the intimate moments with her ex-lover ("There we are again in the middle of the night/ Dancing around the kitchen in the refrigerator light")[37] and the mundane details, such as photo albums, a twin-sized bed, and stopping at red lights.[23][41] After each refrain, the narrator insists that those moments mattered ("I was there, I remember it all too well").[6][37]

The bridge is where the track reaches its climax; Swift sings in her upper register[6][41] and almost shouts with anger ("You call me up again, just to break me like a promise/ So casually cruel in the name of being honest").[42] Her narrator questions why the relationship fell apart, pondering whether it was because she "asked for too much".[43] In the final verse, the narrator mentions the scarf previously detailed in the first verse ("But you keep my old scarf from that very first week/ 'Cause it reminds you of innocence and it smells like me"),[37][40] asserting that the love she gave and felt was real and that the ex-partner must have felt the same.[43]

Brittany Spanos of Rolling Stone wrote that the song describes "the pain of having to piece one's self back together again" after a crumbled relationship.[44] Brad Nelson, in his reviews for The Atlantic and Pitchfork, said that the scarf imagery in the lyrics acts like a Chekhov's gun, symbolizing the persisting emotional flame of the romance long after it has physically ended.[37][40] Nelson compared Swift's lyrics mentioning various memories in a loosely chronological timeline to the technique used in Bob Dylan's "Tangled Up in Blue" (1975)[37] and wrote that it displayed "ambiguity" in Swift's songwriting about failed relationships by not blaming any sides and instead exploring the dissolution of a romance.[40] Meanwhile, J. English of NPR thought that Swift's character in "All Too Well" both puts the balme on her "callous" ex-lover and "owns up to her naivete" for the loss of her innocence, departing from the one-sided blame on past songs such as "Dear John" (2010).[45]
Critical reviews

In reviews of Red, critics lauded "All Too Well" for its detail-heavy lyricism[46] and deemed it the album's centerpiece.[1][27] Dolan,[33] Robbins,[29] and Semon said that the imagery of the couple "dancing around the kitchen in the refrigerator light" was a highlight.[31] Many reviewers picked "All Too Well" as the album's best track, including Jonathan Keefe for Slant Magazine,[24] Sam Lansky for Billboard,[47] Grady Smith for Entertainment Weekly,[34] and Ben Rayner for the Toronto Star.[39] Rosen wrote that it "takes a special songwriter to craft a sneering kiss-off that's also tender valediction", highlighting the bitter accusations ("You call me up again just to break me like a promise/ So casually cruel in the name of being honest") and the affectionate memories ("Photo album on the counter, your cheeks were turning red/ You used to be a little kid with glasses in a twin-size bed").[27]

Some critics highlighted Swift's evolved songwriting. Nelson, in his review for The Atlantic, complimented how Swift expressed "ambiguity" and said that the scarf imagery, like a Chekov's gun, makes it "an exhilarating piece of writing". He added that "All Too Well" explores the dissolution of a romance "so delicately" that he found himself contemplating the song similar to how he would do to a Leonard Cohen song.[40] Keefe highlighted the bridge for featuring one of Swift's "best-ever lines" and how "the song explodes into a full-on bloodletting",[24] American Songwriter's Jewly Hight lauded the portrayal of heartbreak using "tangible details",[48] and PopMatters's Arnold Pan said the song shows how Swift "has amped things up" by "combining the kind of drama that has come natural to her songwriting with a widescreen guitar-driven approach".[49]

Other reviews also praised the track's sound. Robbins wrote that "All Too Well" is one of the Red tracks that "go down like pop punch spiked by pros".[29] Semon described the song as an "acoustic-tinged tearjerker" that has "an alarming intimacy and intense urgency that might even take [Swift's] diehard fans off-guard",[31] and The New York Times Jon Caramanica said that the song "swells until it erupts".[42] Newsday hailed "All Too Well" as the album's highlight "both in drama and execution, with Swift's vocals at their most emotional and her lyrics at their sharpest".[32] In NME, Lucy Harbron selected its bridge as Swift's best and wrote that it was an "ultimate post-breakup big euphoric cry soundtrack".[50]
Charts
2012 weekly chart performance for "All Too Well" Chart (2012) 	Peak
position
Canada (Canadian Hot 100)[17] 	59
US Billboard Hot 100[13] 	80
US Hot Country Songs (Billboard)[14] 	17
US Country Airplay (Billboard)[15] 	58
2021 weekly chart performance for "All Too Well" Chart (2021) 	Peak
position
Austria (Ö3 Austria Top 40)[51] 	16
Belgium (Ultratop 50 Flanders)[52] 	49
Germany (GfK)[53] 	36
Portugal (AFP)[54] 	4
Switzerland (Schweizer Hitparade)[55] 	19
Certifications
Certifications for "All Too Well" Region 	Certification 	Certified units/sales
Australia (ARIA)[18] 	Platinum 	70,000‡
Brazil (Pro-Música Brasil)[19] 	Gold 	30,000‡
Portugal (AFP)[20] 	Gold 	10,000‡
New Zealand (RMNZ)[56] 	Gold 	15,000‡
United Kingdom (BPI)[21] 	Silver 	200,000‡
United States (RIAA)[16] 	Gold 	500,000‡

‡ Sales+streaming figures based on certification alone.
2021 re-recordings
"All Too Well (10 Minute Version)"
Cover of "All Too Well (10 Minute Version)", showing Swift wearing a red beret and beige trench coat, sitting in a convertible
Promotional single by Taylor Swift
from the album Red (Taylor's Version)
Written	2011
Released	November 12, 2021
Studio	

    Conway (Hollywood)
    Electric Lady (New York)
    Rough Customer (Brooklyn)

Genre	Pop rock
Length	10:13
Label	Republic
Songwriter(s)	

    Taylor Swift Liz Rose

Producer(s)	

    Taylor Swift Jack Antonoff

Lyric video
"All Too Well (10 Minute Version)" on YouTube

Swift departed from Big Machine and signed a new contract with Republic Records in 2018. She began re-recording her first six studio albums in November 2020.[57] The decision followed a 2019 dispute between Swift and the talent manager Scooter Braun, who acquired Big Machine Records, over the masters of Swift's albums that the label had released.[58][59] By re-recording the albums, Swift had full ownership of the new masters, which enabled her to encourage licensing of her re-recorded songs for commercial use in hopes of substituting the Big Machine-owned masters.[60] She denoted the re-recordings with a "Taylor's Version" subtitle.[61]
Release

Red's re-recorded album, Red (Taylor's Version), was released on November 12, 2021. It features two versions of "All Too Well": a re-recording of the original album version subtitled "Taylor's Version", and an unabridged version subtitled "10 Minute Version (Taylor's Version) (From the Vault)".[62][63] Speaking on The Tonight Show Starring Jimmy Fallon a day prior to the album's release, Swift said that the 10-minute version was recorded by the sound guy at a Speak Now World Tour rehearsal, and her mother collected a CD recording of it.[64]

Swift wrote and directed All Too Well: The Short Film based on the premise of "All Too Well (10 Minute Version)". It stars Sadie Sink and Dylan O'Brien as a couple in a romantic relationship that ultimately falls apart. Swift premiered the short film at a fan event in New York City on November 12, 2021, the same day that the re-recorded album was released; she also gave a surprise performance of the song there.[65]

"All Too Well (10 Minute Version)" was released via Swift's online store on November 15, 2021, exclusively to US customers.[66] The acoustic "Sad Girl Autumn" version, recorded at Aaron Dessner's Long Pond Studio in the Hudson Valley, was released on November 17.[67] On June 11, 2022, the version used in the short film and the live performance at the premiere were made available for download and streaming, the same day Swift made an appearance at the Tribeca Film Festival to discuss her approach to making the film.[68][69][70]
Lyrics
"All Too Well (10 Minute Version)"
Duration: 31 seconds.0:31
A sample of the final verse of "All Too Well (10 Minute Version)", showing the atmospheric pop rock production with synths, saxophone, and strings
Problems playing this file? See media help.

"All Too Well (10 Minute Version)" has an atmospheric, contemplative pop rock production by Swift and Jack Antonoff,[71] incorporating synths, saxophone, and strings in the outro.[72] There are three new verses added: between the second verse and the second pre-chorus, after the bridge and before the third verse, and after the final verse. There is also an extra stanza in the second chorus. There is the absence of male backing vocals, replaced with Swift's own backing vocals.[73]

Expanding on the narrative of the original "All Too Well", the 10 minute version traces the entire cycle of a fallen relationship with unsparing details—how the idyllic beginning days of the romance (with the boyfriend "tossing [her] the car keys/ 'Fuck the patriarchy' keychain on the ground")[74] quickly soured as Swift's narrator was treated badly by the ex-boyfriend, how he never told her that he loved her, and how he stood up on her 21st birthday.[75][76] She recalls that the ex-boyfriend was possibly ashamed of the relationship ("And there we are again when nobody had to know/ You kept me like a secret but I kept you like an oath")[76] and used their age difference as an excuse to terminate the relationship, but he continued to date women of her age (And I was never good at telling jokes, but the punchline goes/ 'I'll get older, but your lovers stay my age"), a potential reference to the 1993 film Dazed and Confused.[77][78] The breakup left the narrator breaking down in the bathroom, caught by an unnamed actress.[76] In the outro, Swift's narrator contemplates whether the relationship "main you too", asserts that it was a "sacred prayer" and that both she and the ex-boyfriend "remembered it all too well".[72][76]
Critical reception

"All Too Well (10 Minute Version)" was met with universal critical acclaim,[79] often hailed as the standout track on Red (Taylor's Version) and a career highlight for Swift. Rolling Stone music critic Rob Sheffield lauded the 10-minute version for evoking even more intense emotions than the already sentimental five-minute song: "[It] sums up Swift at her absolute best."[80] Helen Brown of The Independent stated the song is a more feminist proposition with its new lyrics.[81] NME's Hannah Mylrea wrote, at its full intended length, "All Too Well (10 Minute Version)" confirms its place as an "epic", exhibiting proficient storytelling, vocal performance and instrumentation.[82] Beth Kirkbride, writing for Clash, said the "epic" song "will go down in history as one of the best breakup songs ever written."[83] Kate Solomon of i wrote the pain "feels raw" in Swift's voice.[84]

Variety's Chris Willman dubbed the song as Swift's "holy grail", and felt glad the singer did not discard the original lyrics, which turn the song into "a stream-of-consciousness epic ballad", filled with more references and specifics of the song's story line.[85] Reviewing for The Line of Best Fit, Paul Bridgewater opined the 10-minute version is "as disarming as it is fascinating"—"an artefact of [Swift's] songwriting and recording process." He asserted that the song magnifies the truncated version's drama and emotion.[86] Slant Magazine critic Jonathan Keefe stated, while the 5-minute version solely focuses on catharsis from a painful relationship, the 10-minute version "more openly implicates the ex who's responsible for causing that pain", and changes the overall tone of "All Too Well" with its new verses and song structure.[87]

Bobby Olivier of Spin praised the song as "Swift's single finest piece of songwriting".[88] Melissa Reguieri, in her USA Today review, said the song "lopes through encyclopedic lyrics that both bite and wound in their honesty and pain."[89] Sputnikmusic staff critic wrote "All Too Well (10 Minute Version)" is "not a flashy pop song or an endearingly rural slice of country" but simply a raw depiction of Swift's emotions, and concluded that the "towering breakup ballad" represents Swift's habit of "expressing these commonplace emotions in uniquely uncommon ways" in her writing.[90] Lydia Burgham of The New Zealand Herald said the lyrics "paint a vivid picture of an ill-fated romance that cuts deep and captures the universal language of heartbreak."[91] In the words of The New York Times critic Lindsay Zoladz, the song is "quite poignantly, about a young woman's attempt to find retroactive equilibrium in a relationship that was based on a power imbalance that she was not at first able to perceive."[75]

The Guardian writer Laura Snapes also dubbed the song an epic track, "one that eviscerates her slick ex in a series of ever-more climactic verses that never resolve to a chorus, just a shuddered realisation of how vividly she recalls his disregard." Snapes associated the lyric "soldier who's returning half her weight" with Swift's eating disorder, and interpreted it as a nod to the physical manifestations of heartache.[92] Spencer Kornhaber of The Atlantic wrote that "All Too Well (10 Minute Version)" contains more specificity in its lyrics, exuding "both warm nostalgia and cooling disdain", which "mesmerizes as Swift's cadence slips and slides against the steadily pounding beat", rather than cluttering the song as one would expect. Kornhaber also admired its outro and tempo shift.[93] In a less complimentary review, Olivia Horn from Pitchfork felt the song was too long and sprawling, undermining the emotional climax of "All Too Well" (2012).[94] In agreement, Zoladz preferred the five-minute version, but appreciated the 10-minute version for its "unapologetic messiness" with nuanced messages about female emotions and societal relationships.[75] Reviewing for Vox, Phillip Garret applauded "All Too Well (10 Minute Version)" as a "whirlwind of a song", with the additional verses describing the context of "All Too Well" "in even greater detail than before".[35]

GQ ranked "All Too Well (10 Minute Version)" as the best single of Swift's career, in October 2022.[95] Alternative Press ranked it as the most emo track in Swift's catalog.[96] The song appeared on several year-end rankings of the best songs of 2021.
Select year-end rankings of "All Too Well (10 Minute Version)" Publication 	List 	Rank 	Ref.
Billboard 	The 100 Best Songs of 2021: Staff List 	
5
	[97]
Consequence 	Top 50 Songs of 2021 	
40
	[98]
Genius 	Top 50 Songs of 2021 	
10
	[99]
Insider 	The Best Songs of 2021 	
1
	[100]
The New York Times 	Jon Pareles's Top 25 Best Songs of 2021 	
4
	[101]
NPR Music 	Best 100 Songs of 2021 	
100
	[102]
The Ringer 	The Best Songs of 2021 	
3
	[103]
Rolling Stone 	The 50 Best Songs of 2021 	
2
	[104]
Rob Sheffield's Top 25 Songs of 2021 	
1
	
[105]
Time 	The Best 10 Songs of 2021 	
3
	[106]
Variety 	The Best Songs of 2021 	
1
	[107]
Accolades
Award and nominations for "All Too Well (10 Minute Version)" Year 	Organization 	Award 	Result 	Ref.
2022 	NME Awards 	Best Music Video 	Nominated 	[108]
ADG Excellence in Production Design Awards 	Music Video 	Won 	[109]
iHeartRadio Music Awards 	Best Lyrics 	Won 	[110]
Kids' Choice Awards 	Favorite Song 	Nominated 	[111]
Joox Thailand Music Awards 	International Song of the Year 	Nominated 	[112]
MTV Video Music Awards 	Video of the Year 	Won 	[113]
Best Longform Video 	Won
Best Direction 	Won
Best Cinematography 	Nominated
Best Editing 	Nominated
MTV Europe Music Awards 	Best Video 	Won 	[114]
Best Longform Video 	Won
UK Music Video Awards 	Best Cinematography in a Video 	Nominated 	[115]
American Music Awards 	Favorite Music Video 	Won 	[116]
2023 	Grammy Awards 	Song of the Year[note 2] 	Nominated 	[117]
Best Music Video 	Won
AIMP Nashville Country Awards 	Song of the Year 	Nominated 	[118]
Commercial performance

"All Too Well (Taylor's Version)"[note 3] debuted at number one on the Australian ARIA Singles Chart the same week Red (Taylor's Version) topped the Australian ARIA Albums Chart, earning Swift a fourth "Chart Double".[note 4] It also helped Swift score a "Chart Double" in Ireland, her second number-one song in the country after "Look What You Made Me Do" (2017).[120] The song was Swift's eighth number-one on the Canadian Hot 100,[121] and entered the UK Singles Chart at number three, the longest song to reach the top five in the UK chart history.[122]

On the US Billboard Hot 100, "All Too Well (Taylor's Version)" marked Swift's eighth number-one song. The song became the longest number-one song in chart history, surpassing Don McLean's 1972 song "American Pie", a feat recognized in the Guinness World Records.[123] Topping the Billboard Hot 100 the same week Red (Taylor's Version) topped the Billboard 200, it marked Swift's record-extending third time to debut atop both charts the same week.[note 5] As Swift's 30th top-10 entry, it made her the sixth artist to reach the milestone. It was her fifth song to top Billboard's Streaming Songs and 23rd to top Billboard's Digital Song Sales charts, extending her record as the female musician with the most chart toppers on both.[124] "All Too Well (Taylor's Version)" is Swift's ninth number-one song on Billboard's Hot Country Songs chart,[79] 18th number-one song on Billboard's Country Digital Song Sales chart, and sixth number-one song on Billboard's Country Streaming Songs chart, confirming her status as the artist with the most number-one songs on the latter two.[125] In addition, the song is the first by a solo female artist to enter the Hot 100 and Hot Country Songs charts at the summit simultaneously.[126]
Charts
Weekly charts
Weekly chart positions for "All Too Well (Taylor's Version)"[note 3] Chart (2021–2022) 	Peak
position
Argentina (Argentina Hot 100)[127] 	79
Australia (ARIA)[128] 	1
Canada (Canadian Hot 100)[121] 	1
Czech Republic (Singles Digitál Top 100)[129] 	95
Denmark (Tracklisten)[130] 	33
Euro Digital Song Sales (Billboard)[131] 	5
Global 200 (Billboard)[132] 	1
Greece International (IFPI)[133] 	11
Hungary (Single Top 40)[134] 	17
Hungary (Stream Top 40)[135] 	31
Iceland (Tónlistinn)[136] 	25
India International Singles (IMI)[137] 	5
Ireland (IRMA)[138] 	1
Italy (FIMI)[139] 	69
Lithuania (AGATA)[140] 	39
Malaysia (RIM)[141] 	1
Netherlands (Dutch Top 40)[142] 	32
Netherlands (Single Top 100)[143] 	39
New Zealand (Recorded Music NZ)[144] 	1
Norway (VG-lista)[145] 	30
Philippines (Billboard)[146] 	8
Singapore (RIAS)[147] 	1
Slovakia (Singles Digitál Top 100)[148] 	81
South Africa (RISA)[149] 	13
Spain (PROMUSICAE)[150] 	25
Sweden (Sverigetopplistan)[151] 	47
UK Singles (OCC)[152] 	3
US Billboard Hot 100[153] 	1
US Adult Pop Airplay (Billboard)[154] 	25
US Hot Country Songs (Billboard)[155] 	1
Vietnam Hot 100 (Billboard)[156] 	82
	
Year-end charts
2022 year-end chart performance for "All Too Well (Taylor's Version)" Chart (2022) 	Position
Australia (ARIA)[157] 	98
Canada (Canadian Hot 100)[158] 	47
Global 200 (Billboard)[159] 	57
US Billboard Hot 100[160] 	76
US Hot Country Songs (Billboard)[161] 	15

Certifications
Certifications for "All Too Well (Taylor's Version)" Region 	Certification 	Certified units/sales
Australia (ARIA)[18]
"10 Minute Version" 	3× Platinum 	210,000‡
Brazil (Pro-Música Brasil)[19] 	Gold 	20,000‡
Brazil (Pro-Música Brasil)[19]
"10 Minute Version" 	2× Platinum 	80,000‡
Denmark (IFPI Danmark)[162]
"10 Minute Version" 	Gold 	45,000‡
Italy (FIMI)[163] 	Gold 	50,000‡
New Zealand (RMNZ)[164] 	2× Platinum 	60,000‡
Poland (ZPAV)[165]
"10 Minute Version" 	Gold 	25,000‡
Spain (PROMUSICAE)[166] 	Platinum 	60,000‡
United Kingdom (BPI)[167] 	Platinum 	600,000‡

‡ Sales+streaming figures based on certification alone.
Release history
Release dates and formats for "All Too Well (Taylor's Version)" Region 	Date 	Format 	Version 	Label(s) 	Ref.
United States 	November 15, 2021 	Digital download 	10-minute version 	Republic 	[66]
Various 	Live acoustic 	[70]
November 17, 2021 	

    Digital downloadstreaming

	Acoustic ("Sad Girl Autumn") 	[168][169]
June 11, 2022 	The Short Film version 	[170]
Live performances
Swift performing "All Too Well" on the Red Tour in 2013 (left) and "All Too Well (10 Minute Version)" at the Eras Tour in 2023 (right)

On January 26, 2014, Swift performed "All Too Well" at the 56th Annual Grammy Awards.[171] Wearing a dramatic beaded gown with sequin detailing and a long train streaming out behind her,[172] she sang while playing piano on a low lit stage, before being joined by a live band midway through the performance. Her performance was praised and received a standing ovation.[173][174] Swift's headbanging at the song's climax gained significant media coverage.[172][175][176]

Swift also performed the song live throughout her Red Tour, while playing the piano.[177] On August 21, 2015, Swift performed the song in Los Angeles at the Staples Center, the only time she did so on The 1989 World Tour.[178] On February 4, 2017, Swift performed the song as part of the Super Saturday Night show in Houston.[179] Swift performed an acoustic version of the song on the first show of her Reputation Stadium Tour in Glendale, Arizona on May 8, 2018,[180] the fifth show in Pasadena, California on May 19, 2018,[181] and the last show of the U.S. leg of the tour in Arlington, Texas on October 6, 2018, the latter of which appeared in her Netflix concert film of the same name.[182] In 2019, she performed the song as part of her one-off City of Lover concert in Paris[183] and at a Tiny Desk Concert for NPR Music.[184]

She performed the 10-minute song after the screening of All Too Well: The Short Film at its film premiere, and on Saturday Night Live the following night.[185][186] Hits dubbed the SNL performance "one of the most riveting musical moments of the year."[187] She also performed it at the 2022 Nashville Songwriter Awards.[188] The 10-minute song is on the regular set list of Swift's sixth headlining concert tour, the Eras Tour (2023–2024).[189] Actors Ryan Gosling and Emily Blunt performed a parody version of the song, inspired by Barbenheimer, on an episode of Saturday Night Live in April 2024.[190]
Impact and legacy

Often dubbed as Swift's magnum opus,[191][192] "All Too Well" has been hailed by music critics, fans and journalists as the best song in Swift's discography, citing the vivid songwriting that evokes deep emotional engagement.[193][75][194] Sheffield commented, "No other song does such a stellar job of showing off her ability to blow up a trivial little detail into a legendary heartache."[195]
Listicles

The song featured on many publications' lists of the best songs from the 2010s decade, including Rolling Stone (5th),[196] Uproxx (10th),[197] Stereogum (24th),[198] and Pitchfork (57th).[199] It featured in unranked 2010s-decade-end lists by Time[200] and Parade,[201] and at number 13 on NPR's readers' poll for the best songs of the same decade.[202] Sheffield ranked "All Too Well" first on his 2010s-decade-end list.[203] Rolling Stone placed "All Too Well" at number 29 of its 2018 list of the 100 Greatest Songs of the Century So Far,[204] and 69 on its list 2021 revision of the 500 Greatest Songs of All Time.[205] In 2021, Sheffield placed the song at number one on his ranking of 206 Taylor Swift songs.[206] In a list titled "The 25 Musical Moments That Defined the First Quarter of the 2020s", Billboard described "All Too Well (10 Minute Version)" as the "crown jewel" of Red (Taylor's Version) and one of 2021's "biggest cultural hits".[207] In 2022, Billboard named "All Too Well (10-Minute Version)" the best breakup song of all time.[208]
Recognition

Critics often regard "All Too Well" as Swift's best song.[209] Billboard stated "All Too Well" is the song that "proved to skeptics who might've thoughtlessly dismissed Swift as a frivolous pop star—in an era when such artists still weren't given nearly as much credit or attention by critics and older music fans as they are now—that she was in fact a truly formidable singer-songwriter." Bruce Warren, assistant general manager for programming for Philadelphia public radio station WXPN, stated that "All Too Well" foreshadowed Swift's music direction for 2020. He said "In 2014 or 2015, you wouldn't have been able to say, '[Taylor Swift] is working with Justin Vernon,' right? ['All Too Well'] foreshadowed the place she's in now... 'All Too Well' showed the potential of how great a songwriter she would be, and how she would evolve as a songwriter. And [Folklore and Evermore] took her to another level."[210] At the 65th Annual Grammy Awards, "All Too Well (10 Minute Version)" lost Song of the Year to Bonnie Raitt's "Just Like That" (2022), garnering controversy over the fact that Swift, who is often considered as one of the foremost songwriters of the 21st-century, has never won Song of the Year despite it being the sixth nomination of her career and "All Too Well" dubbed her best work.[211][212] In March 2023, Stanford University launched an academic course titled "ITALIC 99: All Too Well (Ten Week Version)"; it is "an in-depth analysis" of the song, recognizing Swift's songwriting prowess and related literature.[213]

"All Too Well" is a fan-favorite.[214] Over time, the song achieved a cult following within Swift's fanbase and music critics.[9][23] It is one of Swift's most covered songs.[23] Swift herself remarked this unexpected popularity during her Reputation Stadium Tour:[215]

    It's weird because I feel like this song has two lives to it in my brain. In my brain, there's the life of this song, where this song was born out of catharsis and venting and trying to get over something and trying to understand it and process it. And then there's the life where it went out into the world and you turned this song into something completely different for me. You turned this song into a collage of memories of watching you scream the words to this song, or seeing pictures that you post to me of you having written words to this song in your diary, or you showing me your wrist, and you have a tattoo of the lyrics to this song underneath your skin. And that is how you have changed the song "All Too Well" for me.
    — Swift, Taylor Swift: Reputation Stadium Tour on Netflix[216]

Upon announcement of the release of the original, 10-minute version of "All Too Well" as part of Swift's second re-recorded album, Red (Taylor's Version), the extended version became the most anticipated song from the album.[210] The multimedia release of the album, the 10-minute song, and All Too Well: The Short Film has been described as one of the biggest pop culture moments of 2021.[217][218][219][220]
The scarf

Over the years, the whereabouts of the scarf referenced in the lyrics have become a subject of media attention and speculation.[221][222][223] Amelia Morris, an academic in media and communications, wrote that the scarf became an important part of the "Swiftian mythology", particularly the "Red mythology", that drew the obsession of Swift's fans. According to Morris, this fan debate around the scarf contributed to the release of the "(All Too Well) 10 Minute Version" and the short film, representing an evolution in the song's meaning to Swift—who had expressed how performing the song was difficult for her due to its deeply personal content, and her fans—who related to the song on their own terms.[224]

According to media outlets, the scarf mentioned in the lyrics was originally lost at the residence of American actress Maggie Gyllenhaal, sister of Jake Gyllenhaal.[225] According to The Cut, it is a "very 2008 Americana chic" dark blue scarf with red and gray stripes.[222] Insider confirmed Gucci as the scarf's brand and that Swift was wearing the scarf when she was taking a stroll in London with both Jake and Maggie Gyllenhaal, as seen in multiple photographs by paparazzi.[226][227] Brad Nelson wrote in The Atlantic that the scarf is a Chekhov's gun whose reappearance in the final verse is thoughtful and "brutal". He explained the missing scarf quickly became a "fantastic pop culture mystery" that has created much online buzz.[36]

When asked about the scarf in 2017 by American host Andy Cohen, Maggie Gyllenhaal stated she has no idea where the scarf is, and did not understand why people asked her about it until an interviewer explained the lyrics to her.[225] The scarf's existence or its lyrical use as just a metaphor has been a topic of debate among fans, music critics and pop culture commentators, who "agree it's more than a simple piece of outerwear."[226] According to Sheffield, both the song and the scarf are so significant to Swift's discography that it "should be in the Rock and Roll Hall of Fame."[228] The scarf has become a symbol in Swift's fandom, inspiring jokes, memes, and interview questions.[229] It has even inspired numerous fan-fictions in other fandoms. Writer Kaitlyn Tiffany of The Verge described the scarf as "the green dock light of our time."[230] Insider's Callie Ahlgrim called it a "fabled accessory" and "a source of cultural curiosity".[226] NME critic Rhian Daly said the scarf is "an unlikely pop culture icon in an inanimate object".[231] USA Today said the scarf "re-entered the pop culture conversation" after All Too Well: The Short Film.[232] Kate Leaver of The Sydney Morning Herald wrote only Swift "could make a decade-old item of clothing a universal symbol for heartbreak."[227] The Guardian named the song one of "the most debated lyric mysteries ever".[233]

In 2021, following the release of the film and Red (Taylor's Version), the Google searches for "Taylor Swift red scarf meaning" spiked by 1,400 percent.[234] Fans believe that the scarf is a metaphor for Swift's virginity.[235][236] The scarf has been depicted as a plain red scarf in All Too Well: Short Film and the music video for the country single "I Bet You Think About Me"; in the film, Swift is seen hanging the scarf over a banister, whereas in the latter, she gifts the scarf to a bride, leaving the groom confused.[237][238] Replicas of this scarf—named "The All Too Well Knit Scarf"—were sold on Swift's website.[239] In an interview at the 2022 Toronto International Film Festival, Swift described the red scarf as a metaphor, and that she made it a red-colored scarf to agree with the album's theme.[240]
Personnel

Credits are adapted from the liner notes of Red[11] and Red (Taylor's Version).[241]

"All Too Well" (2012)

    Taylor Swift – vocals, songwriting, production
    Liz Rose – songwriting
    Nathan Chapman – production, acoustic guitar, electric guitar, bass, keyboards, drums, backing vocals, engineering
    LeAnn "Goddess" Bennet – production coordinator
    Drew Bollman – assistant mixer
    Jason Campbell – production coordinator
    Mike "Frog" Griffith – production coordinator
    Brian David Willis – assistant engineer
    Hank Williams – mastering
    Justin Niebank – mixing

"All Too Well (Taylor's Version)" (2021)

    Taylor Swift – lead vocals, songwriter, producer
    Liz Rose – songwriter
    Christopher Rowe – producer, vocal engineer
    David Payne – recording engineer
    Dan Burns – additional engineer
    Austin Brown – assistant engineer, assistant editor
    Bryce Bordone – engineer
    Derek Garten – engineer
    Şerban Ghenea – mixer
    Mike Meadows – acoustic guitar, background vocals
    Amos Heller – bass guitar, synth bass
    Matt Billingslea – drums
    Paul Sidoti – electric guitar
    David Cook – piano
    Max Bernstein – synthesizers

"All Too Well (10 Minute Version) (Taylor's Version) (From the Vault)" (2021)

    Taylor Swift – lead vocals, songwriter, producer
    Liz Rose – songwriter
    Jack Antonoff – producer, recording engineer, engineer, acoustic guitar, bass, electric guitar, keyboards, mellotron, slide guitar, drums, percussion
    Lauren Marquez – assistant recording engineer
    John Rooney – assistant recording engineer
    Jon Sher – assistant recording engineer
    David Hart – engineer, recording engineer: celesta, Hammond B3, piano, reed organ, baritone guitar, Wurlitzer electric piano
    Mikey Freedom Hart – engineer, celesta, Hammond B3, piano, reed organ, baritone guitar, Wurlitzer electric piano
    Sean Hutchinson – engineer, drums, percussion, recording engineer (percussion, drums)
    Jon Gautier – engineer, recording engineer (strings)
    Christopher Rowe – vocal engineer
    Laura Sisk – engineer, recording engineer
    Evan Smith – flutes, saxophone, synthesizers, recording engineer (flutes, saxophone, synthesizers)
    Bryce Bordone – engineer
    Michael Riddleberger – engineer, percussion
    John Rooney – engineer
    Şerban Ghenea – mixer
    Bobby Hawk – strings

Notes

The "10 Minute Version" is officially titled "All Too Well (10 Minute Version) (Taylor's Version) (From the Vault)"
Nominated as "All Too Well (10 Minute Version) (Taylor's Version) (The Short Film)"
Combined chart statistics for both "All Too Well (Taylor's Version)" and "All Too Well (10 Minute Version)".
Swift had achieved a "Chart Double" in 2014 (with 1989 and "Blank Space"), 2020 (twice; with Folklore and "Cardigan", and Evermore and "Willow").[119]

    After "Cardigan" and Folklore, and "Willow" and Evermore, both in 2020.

References

Bernstein, Jonathan (November 18, 2020). "500 Greatest Albums: Taylor Swift Looks Back on Her 'Only True Breakup Album' Red". Rolling Stone. Archived from the original on December 4, 2020. Retrieved July 3, 2021.
Mansfield, Brian (October 17, 2012). "Taylor Swift sees Red all over". USA Today. Archived from the original on January 27, 2013. Retrieved May 21, 2019.
Swift, Taylor (August 23, 2019). Lover Deluxe (Media notes) (1st ed.).
Swift, Taylor (August 23, 2019). Lover Deluxe (Media notes) (4th ed.).
Willman, Chris (August 15, 2014). "Swift Collaboration: Liz Rose Reveals Secrets Behind Taylor's Early Hits". Yahoo! Music. Yahoo!. Archived from the original on December 11, 2017. Retrieved March 5, 2016.
Jones, Nate (May 20, 2024). "Taylor Swift Songs, Ranked From Worst to Best". Vulture. Archived from the original on September 13, 2019. Retrieved August 15, 2024.
Leahey, Andrew (October 24, 2014). "Liz Rose Sets the Scene for Writing With Taylor Swift". Rolling Stone. Archived from the original on July 9, 2021. Retrieved June 30, 2021.
"Taylor Swift Reveals 'All Too Well' Hard To Write—About Jake Gyllenhaal?". Hollywood Life. October 22, 2012. Archived from the original on August 5, 2017. Retrieved April 28, 2013.
Skinner, Paige (February 6, 2019). "From Irving to Nashville to a Grammy: Songwriter Liz Rose Crushes It". Dallas Observer. Archived from the original on July 30, 2019. Retrieved May 21, 2019.
Aniftos, Rania (November 17, 2020). "Yes, Taylor Swift Recorded a 10-Minute Version of 'All Too Well' (With a Swear Word)". Billboard. Archived from the original on September 13, 2021. Retrieved June 29, 2021.
Swift, Taylor (2012). Red (CD album liner notes). Nashville: Big Machine Records / Universal Music Group. 0602537173051.
Lewis, Randy (October 30, 2012). "Taylor Swift raises the bar with a savvy Red marketing campaign". Los Angeles Times. Archived from the original on December 28, 2020. Retrieved August 20, 2024.
"Taylor Swift Chart History (Hot 100)". Billboard. Retrieved October 6, 2021.
"Taylor Swift Chart History (Hot Country Songs)". Billboard. Retrieved October 6, 2021.
"Taylor Swift Chart History (Country Airplay)". Billboard. Retrieved October 6, 2021.
"American single certifications – Taylor Swift – All Too Well". Recording Industry Association of America. Retrieved July 23, 2018.
"Taylor Swift Chart History (Canadian Hot 100)". Billboard. Retrieved October 6, 2021.
"ARIA Charts – Accreditations – 2024 Singles" (PDF). Australian Recording Industry Association. Retrieved June 14, 2024.
"Brazilian single certifications – Taylor Swift – All Too Well (Taylor's Version)" (in Portuguese). Pro-Música Brasil. Retrieved July 24, 2024.
"Portuguese single certifications – Taylor Swift – All Too Well" (PDF) (in Portuguese). Associação Fonográfica Portuguesa. Retrieved May 17, 2022.
"British single certifications – Taylor Swift – All Too Well". British Phonographic Industry. Retrieved May 13, 2022.
Spanos, Brittany (June 9, 2017). "Taylor Swift: 10 Great Deep Cuts You Can Stream Now". Rolling Stone. Archived from the original on September 21, 2022. Retrieved August 15, 2024.
Braca, Nina (November 10, 2021). "Taylor Swift's 'All Too Well': How the Red Fan Favorite Became One of Her Biggest & Most Important Songs". Billboard. Retrieved March 2, 2023.
Keefe, Jonathan (October 22, 2012). "Review: Taylor Swift, Red". Slant Magazine. Archived from the original on October 12, 2013. Retrieved February 25, 2021.
Farley, Rebecca (November 10, 2017). "This Is The Best Taylor Swift Song, No Arguing About It". Refinery29. Archived from the original on September 13, 2021. Retrieved July 2, 2021.
Perone 2017, p. 46.
Rosen, Jody (November 25, 2013). "Why Taylor Swift Is the Reigning Queen of Pop". Vulture. Archived from the original on November 19, 2013. Retrieved August 28, 2024.
Perusse, Bernard (October 23, 2012). "Hot gossip, stale pop; Taylor Swift's tabloid tales lack sharp musical edge". Edmonton Journal. p. C3. ProQuest 1115147198.
Robbins, Michael (October 25, 2012). "Taylor Swift, Red (Big Machine)". Spin. Archived from the original on October 25, 2022. Retrieved August 15, 2024.
"Taylor Swift, Red: Track-By-Track Review". Billboard. October 19, 2012. Archived from the original on February 3, 2018. Retrieved July 3, 2021.
Semon, Craig S. (November 29, 2012). "Taylor Swift seeing Red on new album". Telegram & Gazette. p. 12. ProQuest 1220768531.
"Swift's Red shows she's more than country". Newsday. October 28, 2012. ProQuest 1115521308.
Dolan, Jon (October 23, 2010). "Taylor Swift Red Album Review". Rolling Stone. Archived from the original on January 15, 2018. Retrieved July 3, 2021.
Smith, Grady (October 23, 2012). "What's the best song on Taylor Swift's Red?". Entertainment Weekly. Retrieved March 5, 2024.
Garrett, Philip (November 13, 2021). "Red (Taylor's Version) review: Swift reimagines a modern masterpiece". Vox Magazine. Retrieved March 5, 2024.
Tiffany, Kaitlyn (October 17, 2021). "With fall comes the return of a fantastic pop culture mystery". The Verge. Archived from the original on December 14, 2019. Retrieved February 19, 2021.
Nelson, Brad (August 19, 2019). "Taylor Swift: Red". Pitchfork. Archived from the original on August 20, 2019. Retrieved July 3, 2021.
Gilligan, Eilish (November 4, 2021). "Unpacking 'All Too Well', Taylor Swift's Finest Song". Junkee. Archived from the original on November 4, 2021. Retrieved November 5, 2021.
Rayner, Ben (October 23, 2012). "Pick of the Week: Taylor Swift's Red". Toronto Star. p. E3. ProQuest 1114330580.
Nelson, Brad (November 1, 2012). "If You Listen Closely, Taylor Swift Is Kind of Like Leonard Cohen". The Atlantic. Archived from the original on February 25, 2021. Retrieved August 15, 2024.
Cooney, Samantha; Gutterman, Annabel; Mendez, Moises II; Sonis, Rachel (April 16, 2024). "Taylor Swift's Best Bridges, Ranked". Time. Retrieved August 15, 2024.
Caramanica, Jon (October 24, 2012). "No More Kid Stuff for Taylor Swift". The New York Times. Archived from the original on November 16, 2022. Retrieved July 3, 2021.
Jagota, Vrinda (November 12, 2021). "Taylor Swift: 'All Too Well (10 Minute Version) (Taylor's Version) (From the Vault)'". Pitchfork. Retrieved January 2, 2025.
Spanos, Brittany (August 22, 2016). "Ex-Factor: Taylor Swift's Best Songs About Former Boyfriends". Rolling Stone. Archived from the original on July 17, 2018. Retrieved February 26, 2021.
English, J. (August 28, 2017). "Taylor Swift's Red, A Canonical Coming-Of-Age Album". NPR. Archived from the original on April 12, 2021. Retrieved February 26, 2021.
Vulpo, Mike (August 23, 2019). "Taylor Swift's Original 'All Too Well' Lyrics Revealed in Lover Deluxe Edition". E! Online. Archived from the original on September 13, 2021. Retrieved June 30, 2021.
Lansky, Sam (November 8, 2017). "Why Taylor Swift's Red Is Her Best Album". Billboard. Archived from the original on October 30, 2022. Retrieved August 16, 2024.
Hight, Jewly (October 26, 2012). "Taylor Swift: Red". American Songwriter. Archived from the original on July 26, 2021. Retrieved August 16, 2024.
Pan, Arnold (October 30, 2012). "Taylor Swift: Red". PopMatters. Archived from the original on September 2, 2022. Retrieved August 16, 2024.
Harbron, Lucy (December 1, 2020). "Taylor Swift's infamous 'track fives' – ranked in order of greatness". NME. Archived from the original on September 13, 2021. Retrieved July 3, 2021.
"Taylor Swift – All Too Well" (in German). Ö3 Austria Top 40. Retrieved November 23, 2021.
"Taylor Swift – All Too Well" (in Dutch). Ultratop 50. Retrieved November 28, 2021.
"Taylor Swift – All Too Well" (in German). GfK Entertainment charts. Retrieved November 19, 2021.
"Taylor Swift – All Too Well". AFP Top 100 Singles. Retrieved December 6, 2021.
"Taylor Swift – All Too Well". Swiss Singles Chart. Retrieved November 21, 2021.
"New Zealand single certifications – Taylor Swift – All Too Well". Radioscope. Retrieved December 19, 2024. Type All Too Well in the "Search:" field.
Melas, Chloe (November 16, 2020). "Taylor Swift Speaks Out About Sale of Her Masters". CNN. Archived from the original on November 18, 2020. Retrieved November 19, 2020.
"Taylor Swift Wants to Re-Record Her Old Hits". BBC News. August 22, 2019. Archived from the original on August 22, 2019. Retrieved August 22, 2019.
Finnis, Alex (November 17, 2020). "Taylor Swift Masters: The Controversy around Scooter Braun Selling the Rights to Her Old Music Explained". i. Archived from the original on February 12, 2021. Retrieved February 13, 2021.
Shah, Neil (April 9, 2021). "Taylor Swift Releases New Fearless Album, Reclaiming Her Back Catalog". The Wall Street Journal. Archived from the original on October 8, 2021. Retrieved September 25, 2022.
Espada, Mariah (July 6, 2023). "Taylor Swift Is Halfway Through Her Rerecording Project. It's Paid Off Big Time". Time. Archived from the original on October 27, 2023. Retrieved November 6, 2023.
Al-Heeti, Abrar (November 11, 2021). "Red (Taylor's Version): Release date, tracklist, why Taylor Swift is rerecording her albums". CNET. Archived from the original on November 13, 2021. Retrieved November 13, 2021.
Harris, LaTesha (November 12, 2021). "Taylor Swift, 'All Too Well (10 Minute Version) (Taylor's Version) (From The Vault)'". NPR. Archived from the original on November 15, 2021. Retrieved November 15, 2021.
White, Abbey (November 12, 2021). "Taylor Swift Reveals Story of 'All Too Well' 10-Minute Version". The Hollywood Reporter. Archived from the original on February 18, 2022.
Lipshutz, Jason (November 12, 2021). "Inside Taylor Swift's All Too Well Short Film Premiere: Sobbing, Sing-Alongs & A Surprise Performance". Billboard. Retrieved January 2, 2025.
"All Too Well (10 Minute Version) (Taylor's Version) (From The Vault) Digital Single". taylorswift.com. Archived from the original on November 16, 2021. Retrieved November 15, 2021.
Atkinson, Katie (November 17, 2021). "Taylor Swift Releases an Even Sadder Version of 'All Too Well' for 'Sad Girl Autumn': Listen". Billboard. Archived from the original on November 18, 2021. Retrieved November 18, 2021.
Blistein, Jon (May 2, 2022). "Taylor Swift to Talk All Too Well: The Short Film, Approach to Filmmaking at Tribeca Film Festival". Rolling Stone. Archived from the original on June 11, 2022. Retrieved June 11, 2022.
"Taylor Swift Performs, Brings Out Sadie Sink and Dylan O'Brien at Tribeca Film Festival". Stereogum. June 11, 2022. Archived from the original on November 18, 2022. Retrieved November 18, 2022.
"All Too Well (10 Minute Version) (Taylor's Version) [Live Acoustic] – Single by Taylor Swift on Apple Music". Apple Music. November 15, 2021. Archived from the original on November 16, 2021. Retrieved November 15, 2021.
"Taylor Swift and Jack Antonoff's 20 Best Collaborations". Slant Magazine. November 6, 2022. Archived from the original on March 2, 2023. Retrieved March 2, 2023.
Carson, Sarah (November 12, 2021). "All Too Well (10 Minute Version): did Taylor Swift's greatest song just get better?". i. Retrieved January 2, 2025.
"What Taylor Swift's All Too Well 10 Minute Version Lyrics Mean". Screen Rant. November 16, 2021. Archived from the original on February 3, 2022. Retrieved February 3, 2022.
Gajjar, Saloni (November 12, 2021). "'All Too Well (10 Minute Version)' is the emotional nucleus of Red (Taylor's Version)". The A.V. Club. Archived from the original on March 2, 2023. Retrieved March 2, 2023.
Zoladz, Lindsay (November 15, 2021). "Taylor Swift's 'All Too Well' and the Weaponization of Memory". The New York Times. Archived from the original on November 15, 2021. Retrieved November 15, 2021.
Huff, Lauren (November 12, 2021). "Breaking down the best new verses on Taylor Swift's 10-minute version of 'All Too Well'". Entertainment Weekly. Retrieved January 2, 2025.
Gularte, Alejandra (November 12, 2021). "'All Too Well (10-Minute Version)' Is Everything We Hoped For". Vulture. Archived from the original on December 26, 2021. Retrieved December 26, 2021.
Atkinson, Katie (November 12, 2021). "Taylor Swift's Original 'All Too Well' Lyrics vs. 10-Minute Version: What's New?". Billboard. Retrieved January 2, 2025.
Trust, Gary (November 22, 2021). "Taylor Swift's 'All Too Well (Taylor's Version)' Soars In at No. 1 on Billboard Hot 100". Billboard. Archived from the original on November 23, 2021. Retrieved November 23, 2021.
Sheffield, Rob (November 12, 2021). "Red (Taylor's Version) Makes a Classic Even Better". Rolling Stone. Archived from the original on November 12, 2021. Retrieved November 12, 2021.
Brown, Helen (November 12, 2021). "Taylor Swift's Red is a better, brighter version of a terrific pop album". The Independent. Archived from the original on November 12, 2021. Retrieved November 12, 2021.
Mylrea, Hannah (November 12, 2021). "Taylor Swift – Red (Taylor's Version) review: a retread of heartbreak". NME. Archived from the original on November 12, 2021. Retrieved November 12, 2021.
Kirlbride, Beth (November 12, 2021). "Taylor Swift – Red (Taylor's Version)". Clash. Archived from the original on November 12, 2021. Retrieved November 12, 2021.
Solomon, Kate (November 12, 2021). "Taylor Swift, Red (Taylor's Version), review: How brilliant she is when her heart is in tatters". i. Archived from the original on November 12, 2021. Retrieved November 13, 2021.
Willman, Chris (November 12, 2021). "On Red (Taylor's Version), Taylor Swift's Vault Tracks Are All Too Swell: Album Review". Variety. Archived from the original on November 13, 2021. Retrieved November 16, 2021.
Bridgewater, Paul (November 12, 2021). "Taylor Swift's reworking of Red finds even more magic in her pop blueprint". The Line of Best Fit. Archived from the original on November 12, 2021. Retrieved November 12, 2021.
Keefe, Jonathan (November 17, 2021). "Review: Taylor Swift's Red Redux Flaunts the Singer's Refined Pop Instincts". Slant Magazine. Archived from the original on November 17, 2021. Retrieved November 17, 2021.
Olivier, Bobby (November 12, 2021). "Taylor Swift Remakes Heartbreak Odyssey with Red (Taylor's Version)". Spin. Archived from the original on November 12, 2021. Retrieved November 12, 2021.
Rugieri, Melissa (November 13, 2021). "Taylor Swift's new Red is a fan feast of 30 sensitive, angry and transformative songs". USA Today. Archived from the original on November 12, 2021. Retrieved November 13, 2021.
"Review: Taylor Swift – Red (Taylor's Version)". Sputnikmusic. November 12, 2021. Archived from the original on November 18, 2021. Retrieved November 12, 2021.
Burgham, Lydia (November 13, 2021). "Red (Taylor's Version) Review: Why Red Is Taylor Swift's Magnum Opus". The New Zealand Herald. Archived from the original on November 12, 2021. Retrieved November 13, 2021.
Snapes, Laura (November 12, 2021). "Taylor Swift: Red (Taylor's Version) review – getting back together with a classic". The Guardian. Archived from the original on November 12, 2021. Retrieved November 12, 2021.
Kornhaber, Spencer (November 14, 2021). "On SNL, Taylor Swift Stopped Time". The Atlantic. Archived from the original on November 20, 2021. Retrieved December 5, 2021.
Horn, Olivia (November 15, 2021). "Taylor Swift: Red (Taylor's Version)". Pitchfork. Archived from the original on November 15, 2021. Retrieved November 15, 2021.
Ford, Lucy (October 11, 2022). "Every Taylor Swift single, ranked". British GQ. Archived from the original on December 3, 2022. Retrieved October 14, 2022.
Olivier, Bobby (October 18, 2022). "15 of Taylor Swift's most emo songs ever, ranked". Alternative Press. Archived from the original on October 19, 2022. Retrieved October 19, 2022.
"The 100 Best Songs of 2021: Staff List". Billboard. December 7, 2021. Archived from the original on December 7, 2021. Retrieved December 7, 2021.
"Top 50 Songs of 2021". Consequence. December 6, 2021. Archived from the original on December 6, 2021. Retrieved December 7, 2021.
"The Genius Community's 50 Best Songs Of 2021". Genius. Archived from the original on December 31, 2021. Retrieved January 3, 2022.
Ahlgrim, Callie. "The best songs of 2021". Insider. Archived from the original on September 14, 2022. Retrieved December 11, 2021.
Pareles, Jon; Caramanica, Jon; Zoladz, Lindsay (December 7, 2021). "Best Songs of 2021". The New York Times. ISSN 0362-4331. Archived from the original on November 1, 2022. Retrieved December 7, 2021.
Harris, LaTesha (December 2, 2021). "The Best Music of 2021". NPR Music. Archived from the original on December 9, 2021. Retrieved December 5, 2021.
"The Best Songs of 2021". The Ringer. December 9, 2021. Archived from the original on December 23, 2021. Retrieved December 23, 2021.
Blistein, Jon; Bernstein, Jonathan; Chan, Tim; Conteh, Mankaprr; Dolan, Jon; Dukes, Will; Freeman, Jon; Grow, Kory; Hudak, Joseph; Kwak, Kristine; Leight, Elias; Lopez, Julyssa; Martoccio, Angie; Sheffield, Rob; Reeves, Mosi; Shteamer, Hank; Vozick-Levinson, Simon (December 6, 2021). "The 50 Best Songs of 2021". Rolling Stone. Archived from the original on December 6, 2021. Retrieved December 6, 2021.
Rob, Sheffield. "Rob Sheffield's Top 25 Songs of 2021". Rolling Stone. Archived from the original on December 22, 2021. Retrieved December 22, 2021.
Chow, Andrew R.; Lang, Cady (December 5, 2021). "The 10 Best Songs of 2021". Time. Archived from the original on December 7, 2021. Retrieved December 5, 2021.
Willman, Chris (December 31, 2021). "The 50 Best Songs of 2021". Variety. Archived from the original on July 9, 2022. Retrieved November 23, 2022.
Trendell, Andrew (January 27, 2022). "BandLab NME Awards 2022: Full list of nominations revealed". NME. Archived from the original on January 27, 2022. Retrieved July 26, 2022.
Giardina, Carolyn; Gajewski, Ryan (March 6, 2022). "'Dune,' 'Nightmare Alley,' 'No Time to Die' Win Art Directors Guild Awards". The Hollywood Reporter. Archived from the original on March 6, 2022. Retrieved June 4, 2022.
"2022 iHeartRadio Music Awards: See The Full List Of Winners". iHeart. Archived from the original on April 8, 2022. Retrieved October 13, 2022.
Gajewski, Ryan (April 10, 2022). "Kids' Choice Awards: 'Spider-Man: No Way Home' Wins Big; Dr. Jill Biden Speaks". The Hollywood Reporter. Archived from the original on April 10, 2022. Retrieved October 13, 2022.
"รูปที่ 11/20 จากอัลบั้มรวมรูปภาพของ JOOX Thailand Music Awards 2022 เปิดโหวตแล้ว! พร้อมสุดยอดรางวัลดนตรีแห่งปี 12 สาขา". www.sanook.com/music/ (in Thai). April 29, 2022. Archived from the original on October 13, 2022. Retrieved October 13, 2022.
Montgomery, Daniel (August 29, 2022). "2022 VMA winners list: Here's who won in all categories at MTV Video Music Awards". GoldDerby. Archived from the original on August 29, 2022. Retrieved August 29, 2022.
"See the Complete List of MTV EMA 2022 Nominees". MTV EMA. Archived from the original on October 12, 2022. Retrieved October 12, 2022.
"UK Music Video Awards 2022: all the nominations for this year's UKMVAs". Promonews. September 28, 2022. Archived from the original on January 12, 2023. Retrieved September 28, 2022.
Grein, Paul (October 13, 2022). "Bad Bunny Leads 2022 American Music Awards Nominations: Full List". Billboard. Archived from the original on October 14, 2022. Retrieved October 13, 2022.
Moreau, Jordan (February 5, 2023). "Grammy Winners 2023: Full List (Updating Live)". Variety. Archived from the original on February 6, 2023. Retrieved February 6, 2023.
"2023 NOMINEES". AIMP Nashville Awards. Archived from the original on February 1, 2023. Retrieved February 1, 2023.
Brandle, Lars (November 19, 2021). "Taylor Swift Snags Australian Chart Double". Billboard. Archived from the original on November 19, 2021. Retrieved November 19, 2021.
"Taylor Swift scores Irish charts double". RTÉ. November 19, 2021. Archived from the original on November 20, 2021. Retrieved November 20, 2021.
"Taylor Swift Chart History (Canadian Hot 100)". Billboard. Retrieved November 23, 2021.
"Taylor Swift lands eighth UK number one album with re-recorded Red". BBC News. November 19, 2021. Archived from the original on November 20, 2021. Retrieved November 20, 2021.
"Taylor Swift's 10-minute 'All Too Well' is longest song to reach No.1". Guinness World Records. November 26, 2021. Archived from the original on November 29, 2021. Retrieved November 26, 2021.
Trust, Gary (November 22, 2021). "Taylor Swift's 'All Too Well (Taylor's Version)' Soars In at No. 1 on Billboard Hot 100". Billboard. Archived from the original on November 23, 2021. Retrieved November 26, 2021.
Asker, Jim (November 23, 2021). "Taylor Swift Rules Country Charts With Re-Recorded Red Album & 'All Too Well' Single". Billboard. Archived from the original on December 2, 2021. Retrieved November 26, 2021.
Hussey, Allison (November 22, 2021). "Taylor Swift Sets New Record for Longest No. 1 Song With "All Too Well (10 Minute Version)"". Pitchfork. Archived from the original on November 25, 2021. Retrieved September 1, 2021.
"Taylor Swift – Chart History (Argentina Hot 100)" Billboard Argentina Hot 100 Singles for Taylor Swift. Retrieved November 24, 2021.
"Taylor Swift – All Too Well". ARIA Top 50 Singles. Retrieved October 17, 2022.
"ČNS IFPI" (in Czech). Hitparáda – Digital Top 100 Oficiální. IFPI Czech Republic. Note: Select 46. týden 2021 in the date selector. Retrieved November 22, 2021.
"Track Top-40 Uge 46, 2021". Hitlisten. Archived from the original on November 23, 2021. Retrieved November 24, 2021.
"Taylor Swift Chart History (Euro Digital Song Sales)". Billboard. Retrieved May 17, 2022.
"Taylor Swift Chart History (Global 200)". Billboard. Retrieved November 23, 2021.
"Digital Singles Chart (International)" (in Greek). IFPI Greece. Archived from the original on December 2, 2021. Retrieved November 29, 2021.
"Archívum – Slágerlisták – MAHASZ" (in Hungarian). Single (track) Top 40 lista. Magyar Hanglemezkiadók Szövetsége. Retrieved November 26, 2021.
"Archívum – Slágerlisták – MAHASZ" (in Hungarian). Stream Top 40 slágerlista. Magyar Hanglemezkiadók Szövetsége. Retrieved November 26, 2021.
"Tónlistinn – Lög" [The Music – Songs] (in Icelandic). Plötutíðindi. Archived from the original on November 21, 2021. Retrieved November 21, 2021.
"IMI International Top 20 Singles for week ending 22nd November 2021 | Week 46 of 52". Indian Music Industry. Archived from the original on November 22, 2021. Retrieved November 22, 2021.
"Official Irish Singles Chart Top 50". Official Charts Company. Retrieved November 19, 2021.
"Top Singoli – Classifica settimanale WK 46" (in Italian). Federazione Industria Musicale Italiana. Archived from the original on November 15, 2019. Retrieved November 20, 2021.
"2021 47-os savaitės klausomiausi (Top 100)" (in Lithuanian). AGATA. November 26, 2021. Archived from the original on November 26, 2021. Retrieved November 26, 2021.
"Top 20 Most Streamed International + Domestic Songs Week 46 / (12/11/2021-18/11/2021)". RIM. November 27, 2021. Archived from the original on December 6, 2021. Retrieved November 27, 2021.
"Nederlandse Top 40 – week 49, 2021" (in Dutch). Dutch Top 40. Retrieved December 4, 2021.
"Taylor Swift – All Too Well" (in Dutch). Single Top 100. Retrieved November 20, 2021.
"NZ Top 40 Singles Chart". Recorded Music NZ. November 22, 2021. Archived from the original on November 19, 2021. Retrieved November 20, 2021.
"VG-lista – Topp 20 Single 2021–47". VG-lista. Archived from the original on December 6, 2021. Retrieved November 27, 2021.
"Taylor Swift Chart History (Philippines Songs)". Billboard. Archived from the original on May 17, 2022. Retrieved April 26, 2022.
"RIAS Top Charts Week 46 (12 – 18 Nov 2021)". November 23, 2021. Archived from the original on November 23, 2021. Retrieved November 23, 2021.
"ČNS IFPI". IFPI ČR. Note: Select SK SINGLES DIGITAL TOP 100 and insert 202146 into search. Archived from the original on November 3, 2020. Retrieved November 22, 2021.
"Local & International Streaming Chart Top 100: Week 46". The Official South African Charts. Recording Industry of South Africa. Archived from the original on November 25, 2021. Retrieved November 26, 2021.
"Top 100 Canciones – Semana 46: del 12 November 2021 al 18.11.2021". Productores de Música de España. Archived from the original on November 24, 2021. Retrieved November 23, 2021.
"Veckolista Singlar, vecka 46" (in Swedish). Sverigetopplistan. Archived from the original on November 21, 2021. Retrieved November 19, 2021.
"Official Singles Chart Top 100". Official Charts Company. Retrieved November 19, 2021.
"Taylor Swift Chart History (Hot 100)". Billboard. Retrieved November 22, 2021.
"Taylor Swift Chart History (Adult Pop Songs)". Billboard. Retrieved January 25, 2022.
"Taylor Swift Chart History (Hot Country Songs)". Billboard. Retrieved November 23, 2021.
"Taylor Swift Chart History (Billboard Vietnam Hot 100)". Billboard. Archived from the original on June 6, 2022. Retrieved March 2, 2022.
"ARIA Top 100 Singles Chart for 2022". Australian Recording Industry Association. Archived from the original on January 4, 2023. Retrieved January 4, 2023.
"Canadian Hot 100 – Year-End 2022". Billboard. Archived from the original on December 1, 2022. Retrieved December 2, 2022.
"Billboard Global 200 – Year-End 2022". Billboard. Archived from the original on December 1, 2022. Retrieved December 2, 2022.
"Hot 100 Songs – Year-End 2022". Billboard. Archived from the original on December 1, 2022. Retrieved December 2, 2022.
"Hot Country Songs – Year-End 2022". Billboard. Archived from the original on December 2, 2022. Retrieved December 1, 2022.
"Danish single certifications – Taylor Swift – All Too Well (Taylor's Version)". IFPI Danmark. Retrieved August 28, 2024.
"Italian single certifications – Taylor Swift – All Too Well (Taylor's Version)" (in Italian). Federazione Industria Musicale Italiana. Retrieved May 13, 2024. Select "2024" in the "Anno" drop-down menu. Type "All Too Well (Taylor's Version)" in the "Filtra" field. Select "Singoli" under "Sezione".
"New Zealand single certifications – Taylor Swift – All Too Well (Taylor's Version)". Radioscope. Retrieved December 19, 2024. Type All Too Well (Taylor's Version) in the "Search:" field.
"OLiS - oficjalna lista wyróżnień" (in Polish). Polish Society of the Phonographic Industry. Retrieved August 28, 2024. Click "TYTUŁ" and enter All Too Well (Taylor's Version) in the search box.
"Spanish single certifications – Taylor Swift – All Too Well (Taylor's Version)". El portal de Música. Productores de Música de España. Retrieved April 22, 2024.
"British single certifications – Taylor Swift – All Too Well (Taylor's Version)". British Phonographic Industry. Retrieved May 29, 2023.
"All Too Well (Sad Girl Autumn Version) – Recorded at Long Pond Studios". taylorswift.com. Archived from the original on November 17, 2021. Retrieved November 17, 2021.
"All Too Well (Sad Girl Autumn Version) [Recorded at Long Pond Studios] – Single by Taylor Swift". Apple Music. November 18, 2021. Archived from the original on November 19, 2021. Retrieved November 19, 2021.
"All Too Well (10 Minute Version) [The Short Film] by Taylor Swift". Apple Music. June 11, 2022. Archived from the original on June 11, 2022. Retrieved June 11, 2022.
Rosen, Christopher (January 26, 2014). "Taylor Swift Grammys Performance Of 'All To Well' Is Worth All Feelings". The Huffington Post. Archived from the original on October 11, 2016. Retrieved January 27, 2014.
Wickman, Kase (January 26, 2014). "How Taylor Swift's Grammy Performance Helped Fans Forgive Her Exes". MTV. Archived from the original on January 29, 2014. Retrieved January 27, 2014.
Marecsa, Rachel (January 27, 2014). "Grammys 2014: Taylor Swift reacts too early before losing Album of the Year award to Daft Punk". New York Daily News. Archived from the original on January 27, 2014. Retrieved January 27, 2014.
Harp, Justin (January 27, 2014). "Grammy Awards 2014: Taylor Swift dazzles with 'All Too Well' – video". Digital Spy. Archived from the original on January 30, 2014. Retrieved January 27, 2014.
Wyatt, Daisy (January 27, 2014). "Grammys 2014: Beyonce and Jay Z open bizarre awards featuring Taylor Swift head-banging". The Independent. Archived from the original on January 27, 2014. Retrieved January 27, 2014.
Tucker, Rebecca (January 27, 2014). "'F–k you, Grammys': From Trent Reznor's tweet to Taylor Swift's hair, presenting the best and worst moments from the 2014 Grammy Awards". National Post. Archived from the original on January 27, 2014. Retrieved January 27, 2014.
Sheffield, Rob (October 18, 2013). "Taylor Swift's 'Red' Tour: Her Amps Go Up to 22 | Rob Sheffield". Rolling Stone. Archived from the original on March 26, 2014. Retrieved January 27, 2014.
Ginn, Leighton (August 22, 2015). "Taylor Swift has banner performance during first of five sold-out shows in Los Angeles". Los Angeles Daily News. Archived from the original on March 29, 2016. Retrieved September 3, 2015.
Atkinson, Katie (February 5, 2017). "Taylor Swift Performs 'Better Man' & 'I Don't Wanna Live Forever' for First Time at Stunning Pre-Super Bowl Set". Billboard. Archived from the original on November 21, 2021.
Chiu, Melody (May 9, 2018). "Taylor Swift Kicks Off Her Reputation Stadium Tour in Arizona – Find Out All the Details". People. Archived from the original on November 21, 2021. Retrieved July 3, 2021.
Iasimone, Ashley (May 26, 2018). "All the Surprise Songs Taylor Swift Has Performed on Her Reputation Stadium Tour B-Stage (So Far)". Billboard. Archived from the original on May 27, 2018. Retrieved July 4, 2021.
Warner, Denise (December 31, 2018). "Taylor Swift Knows All Too Well How to Put on a Masterful Performance: Netflix Doc Review". Billboard. Archived from the original on November 21, 2021. Retrieved July 3, 2021.
Mylrea, Hannah (September 10, 2019). "Taylor Swift's The City of Lover concert: a triumphant yet intimate celebration of her fans and career". NME. Archived from the original on September 16, 2019. Retrieved May 20, 2020.
Mamo, Heran (October 11, 2019). "Are You '...Ready For It?' Taylor Swift's Tiny Desk Concert Is About to Drop". Billboard. Archived from the original on October 12, 2019. Retrieved October 12, 2019.
Derschowitz, Jessica; Lamphier, Jason (November 12, 2021). "Taylor Swift debuts 'All Too Well' short film with surprise performance". Entertainment Weekly. Archived from the original on November 13, 2021. Retrieved November 13, 2021.
Taylor Swift – All Too Well (10 Minute Version) (Live on Saturday Night Live), November 14, 2021, retrieved August 21, 2023
Glickman, Lenny Beer and Simon. "GRAMMY CHEW: RUMINATING ON SONG OF THE YEAR". HITS Daily Double. Archived from the original on December 23, 2022. Retrieved December 16, 2022.
Hudak, Joseph (September 21, 2022). "Taylor Swift Reveals New Song Title 'Mastermind,' Sings 10-Minute 'All Too Well' at Surprise Nashville Appearance". Rolling Stone. Archived from the original on September 29, 2022. Retrieved November 22, 2022.
Shafer, Ellise (March 18, 2023). "Taylor Swift Eras Tour: The Full Setlist From Opening Night". Variety. Archived from the original on March 18, 2023. Retrieved March 19, 2023.
Nordyke, Kimberly (April 15, 2024). "Taylor Swift Reacts to Ryan Gosling and Emily Blunt's Cover of "All Too Well" on 'SNL'". The Hollywood Reporter. Retrieved April 15, 2024.
"Who Will Win—and Who Should—at the 2023 Grammy Awards". Time. Archived from the original on February 1, 2023. Retrieved February 1, 2023.
Harris, Latesha (November 12, 2021). "Taylor Swift, 'All Too Well (10 Minute Version) (Taylor's Version) (From The Vault)'". NPR. Archived from the original on November 20, 2021. Retrieved February 1, 2023.
Mylrea, Hannah (September 8, 2020). "Every Taylor Swift song ranked in order of greatness". NME. Archived from the original on September 17, 2020. Retrieved February 19, 2021.
Moran, Meredith (November 12, 2021). "How Taylor Swift's Best Song Went From Underground Fave to 10-Minute Masterpiece". Slate. Archived from the original on November 14, 2021. Retrieved November 15, 2021.
Sheffield, Rob (October 26, 2021). "All 199 of Taylor Swift's Songs". Rolling Stone. Archived from the original on February 15, 2021. Retrieved October 26, 2021.
"The 100 Best Songs of the 2010s". Rolling Stone. Archived from the original on December 26, 2019. Retrieved December 6, 2019.
"All The Best Songs Of The 2010s, Ranked". Uproxx. October 9, 2019. Archived from the original on October 10, 2019. Retrieved December 11, 2019.
"The 200 Best Songs of the 2010s". Stereogum. November 5, 2019. Archived from the original on November 6, 2019. Retrieved November 20, 2019.
"The 200 Best Songs of the 2010s". Pitchfork. October 7, 2019. Archived from the original on October 7, 2019. Retrieved February 2, 2019.
Johnston, Maura (November 18, 2019). "The 10 Best Songs of the 2010s". Time. Archived from the original on September 18, 2020. Retrieved December 11, 2019.
Sager, Jessica (December 27, 2019). "50 Best Songs of the 2010s That We'll Be Listening to For Decades to Come". Parade. Archived from the original on December 27, 2019. Retrieved December 27, 2019.
Hilton, Robin; Boilen, Bob (November 27, 2019). "The 2010s: NPR Listeners Pick Their Top Songs Of The Decade". NPR. Archived from the original on November 28, 2019. Retrieved November 27, 2019.
Sheffield, Rob (December 23, 2019). "Rob Sheffield's 50 Best Songs of the 2010s". Rolling Stone. Archived from the original on October 22, 2020. Retrieved December 24, 2019.
Hoard, Christian; Weingarten, Christopher R.; Dolan, Jon; Leight, Elias; Spanos, Brittany; Exposito, Suzy; Grow, Kory; Grant, Sarah; Vozick-Levinson, Simon; Greene, Andy; Hermes, Will (June 28, 2018). "The 100 Greatest Songs of the Century – So Far". Rolling Stone. Archived from the original on November 23, 2019. Retrieved November 20, 2019.
"The 500 Greatest Songs of All Time". Rolling Stone. September 15, 2021. Archived from the original on September 16, 2021. Retrieved September 15, 2021.
Sheffield, Rob (October 26, 2021). "All 206 of Taylor Swift's Songs, Ranked". Rolling Stone. Archived from the original on October 15, 2020. Retrieved February 6, 2022.
"The 25 Musical Moments That Defined the First Quarter of the 2020s". Billboard. July 5, 2022. Archived from the original on July 5, 2022. Retrieved July 5, 2022.
Dailey, Hannah (July 20, 2022). "50 Best Breakup Songs of All Time". Billboard. Archived from the original on July 26, 2022. Retrieved July 30, 2022.
"The 21 Best Albums Of 2021 (Part One) -". Rolling Stone. December 29, 2021. Archived from the original on January 6, 2022. Retrieved January 6, 2022.
"Taylor Swift's 'All Too Well': How the 'Red' Fan Favorite Became One of Her Biggest & Most Important Songs". Billboard. Archived from the original on November 10, 2021. Retrieved November 11, 2021.
Lewis, Hilary (February 6, 2023). "Grammys: Jay-Z Shut Out as Taylor Swift and Adele Win One Award Each and Bonnie Raitt Surprises". The Hollywood Reporter. Archived from the original on February 7, 2023. Retrieved February 7, 2023.
"Bonnie Raitt Wins Song of the Year for "Just Like That" at 2023 Grammys". Pitchfork. February 6, 2023. Archived from the original on February 7, 2023. Retrieved February 7, 2023.
"Martinez | Stanford needs a course on Taylor Swift's social media marketing". March 7, 2023. Archived from the original on March 8, 2023. Retrieved March 9, 2023.
Wang, Jessica. "Peloton members can now exercise and cry to 'Red (Taylor's Version)'". Entertainment Weekly. Archived from the original on January 6, 2022. Retrieved January 6, 2022.
"Taylor Swift Shares 'All Too Well' Teaser From Netflix Concert Film: 'Moments Like This Defined the Tour'". Billboard. Archived from the original on January 12, 2019. Retrieved September 11, 2019.
Taylor Swift: Reputation Stadium Tour (Movie). Netflix. December 31, 2018. Event occurs at 1:03:32.[permanent dead link]
Andrew, Scottie; Asmelash, Leah (December 29, 2021). "The pop culture moments of 2021 we couldn't forget if we tried". CNN. Archived from the original on December 29, 2021. Retrieved December 30, 2021.
Ruiz, Michelle (December 15, 2021). "The 10 best pop-culture moments of 2021". Vogue India. Archived from the original on December 29, 2021. Retrieved December 30, 2021.
Ruggieri, Melissa. "Ye's 'Donda' rollout, Adele's triumphant return and more of 2021's biggest music moments". USA TODAY. Archived from the original on December 29, 2021. Retrieved December 30, 2021.
"How Taylor Swift reclaimed 2012 to win 2021". Los Angeles Times. December 17, 2021. Archived from the original on December 30, 2021. Retrieved December 30, 2021.
Khomami, Nadia (November 15, 2021). "Where's Taylor Swift's scarf – is it in Jake Gyllenhaal's drawer?". The Guardian. Archived from the original on March 6, 2022. Retrieved March 6, 2022.
Mercado, Mia (November 16, 2021). "Where the Hell Is Taylor Swift's Scarf?". The Cut. Archived from the original on February 3, 2022. Retrieved February 3, 2022.
Yahr, Emily (November 12, 2021). "The story behind Taylor Swift's 10-minute version of 'All Too Well', the song making fans lose their minds". The Washington Post. Archived from the original on November 29, 2022. Retrieved March 2, 2023.
Morris 2024, p. 12.
Lang, Cady (September 13, 2017). "Maggie Gyllenhaal Addresses Taylor Swift's Lost Scarf Lyric". Time. Archived from the original on November 29, 2020. Retrieved February 25, 2021.
Ahlgrim, Callie. "How Taylor Swift's scarf went from an innocent accessory to Jake Gyllenhaal's worst nightmare". Insider. Archived from the original on February 10, 2023. Retrieved February 3, 2022.
"Why is Taylor Swift's scarf all the rage right now?". Sydney Morning Herald. November 18, 2021. Archived from the original on February 3, 2022. Retrieved February 3, 2022.
Sheffield, Rob (November 24, 2020). "All 129 of Taylor Swift's Songs, Ranked by Rob Sheffield". Rolling Stone. Archived from the original on February 15, 2021. Retrieved February 19, 2021.
Lang, Cady (September 14, 2017). "At Last Maggie Gyllenhaal Acknowledges the Most Important Taylor Swift Scarf Rumor". Time. Archived from the original on June 25, 2021. Retrieved May 21, 2019.
Tiffany, Kaitlyn (October 17, 2017). "With fall comes the return of a fantastic pop culture mystery". The Verge. Archived from the original on December 14, 2019. Retrieved May 21, 2019.
Daly, Rhian (November 13, 2021). "Taylor Swift's 'All Too Well' short film highlights the emotional power of her storytelling". NME. Archived from the original on November 13, 2021. Retrieved November 13, 2021.
"2021 was another difficult year. These 100 things made USA TODAY's entertainment team happy". USA TODAY. Archived from the original on December 27, 2021. Retrieved December 28, 2021.
"From American Pie to All Too Well: the most debated lyric mysteries ever". the Guardian. July 28, 2022. Archived from the original on July 30, 2022. Retrieved July 30, 2022.
Flanagan, Hanna (November 23, 2021). "Taylor Swift's Red (Taylor's Version) Causes Google Searches for Red Lipstick and Red Scarves to Spike". People. Archived from the original on November 23, 2021. Retrieved December 29, 2021.
Bonghi, Gabrielle (February 18, 2014). "Jake Gyllenhaal reportedly took Taylor Swift's virginity, and she was a mess about it". www.inquirer.com. Archived from the original on December 19, 2023. Retrieved December 19, 2023.
"Taylor Swift confirms 'All Too Well' red scarf metaphor and fans think it's about her virginity". theedge.co.nz. Retrieved December 19, 2023.
"The Hat! The Ring! Taylor Swift's 'I Bet You Think About Me' Easter Eggs". Us Weekly. November 15, 2021. Archived from the original on February 3, 2022. Retrieved February 3, 2022.
Dellatto, Marisa. "Taylor Swift's Scarf Explained: Why Everyone Wants To Know Where It Is—And If Jake Gyllenhaal Has It". Forbes. Archived from the original on February 3, 2022. Retrieved February 3, 2022.
"The All Too Well Knit Scarf". Taylor Swift Official Store. Archived from the original on December 1, 2021. Retrieved February 3, 2022.
Alex, Hopper (September 10, 2022). "Taylor Swift Addresses the Red Scarf at Toronto International Film Festival—"I'm Just Going to Stop"". American Songwriter. Archived from the original on September 10, 2022. Retrieved September 10, 2022.

    Swift, Taylor (2021). Red (Taylor's Version) (vinyl liner notes). Republic Records.

Sources

    Morris, Amelia (April 4, 2024). "Drew a map on your bedroom ceiling: fandoms, nostalgic girlhood and digital bedroom cultures in the Swiftie-sphere". Celebrity Studies: 1–19. doi:10.1080/19392397.2024.2338540. hdl:10871/137283.
    Perone, James E. (July 14, 2017). "Red". The Words and Music of Taylor Swift. ABC-CILO. pp. 43–54. ISBN 978-1-4408-5295-4.

External links

    Lyrics of this song at Taylor Swift's official site

    vte

Taylor Swift songs

    Singles discography

Taylor Swift	

    "Tim McGraw" "Picture to Burn" "Teardrops on My Guitar" "A Place in This World" "Should've Said No" "Our Song"

Fearless	

    "Fearless" "Fifteen" "Love Story" "Hey Stephen" "White Horse" "You Belong with Me" "Breathe" "You're Not Sorry" "The Way I Loved You" "Forever & Always" "The Best Day" "Change"

Taylor's Version	

    "You All Over Me" "Mr. Perfectly Fine" "That's When"

Speak Now	

    "Mine" "Sparks Fly" "Back to December" "Speak Now" "Dear John" "Mean" "The Story of Us" "Never Grow Up" "Enchanted" "Better than Revenge" "Innocent" "Haunted" "Long Live" "Ours"

Taylor's Version	

    "Electric Touch" "When Emma Falls in Love" "I Can See You" "Castles Crumbling"

Red	

    "State of Grace" "Red" "Treacherous" "I Knew You Were Trouble" "All Too Well" "22" "We Are Never Ever Getting Back Together" "The Last Time" "Holy Ground" "Everything Has Changed" "Begin Again"

Taylor's Version	

    "Nothing New" "Message in a Bottle" "I Bet You Think About Me" "Forever Winter"

1989	

    "Welcome to New York" "Blank Space" "Style" "Out of the Woods" "All You Had to Do Was Stay" "Shake It Off" "I Wish You Would" "Bad Blood" "Wildest Dreams" "How You Get the Girl" "This Love" "I Know Places" "Clean" "You Are in Love" "New Romantics"

Taylor's Version	

    "'Slut!'" "Say Don't Go" "Now That We Don't Talk" "Suburban Legends" "Is It Over Now?"

Reputation	

    "...Ready for It?" "End Game" "I Did Something Bad" "Don't Blame Me" "Delicate" "Look What You Made Me Do" "So It Goes..." "Gorgeous" "Getaway Car" "Dress" "Call It What You Want" "New Year's Day"

Lover	

    "I Forgot That You Existed" "Cruel Summer" "Lover" "The Man" "The Archer" "Miss Americana & the Heartbreak Prince" "Paper Rings" "Cornelia Street" "Death by a Thousand Cuts" "London Boy" "Soon You'll Get Better" "False God" "You Need to Calm Down" "Me!"

Folklore	

    "The 1" "Cardigan" "The Last Great American Dynasty" "Exile" "My Tears Ricochet" "Mirrorball" "Seven" "August" "This Is Me Trying" "Illicit Affairs" "Invisible String" "Mad Woman" "Epiphany" "Betty" "Peace" "Hoax" "The Lakes"

Evermore	

    "Willow" "Champagne Problems" "'Tis the Damn Season" "Tolerate It" "No Body, No Crime" "Happiness" "Dorothea" "Coney Island" "Long Story Short" "Marjorie"

Midnights	

    "Lavender Haze" "Maroon" "Anti-Hero" "Snow on the Beach" "You're on Your Own, Kid" "Midnight Rain" "Question...?" "Vigilante Shit" "Bejeweled" "Labyrinth" "Karma" "Sweet Nothing" "Mastermind" "Hits Different" "Bigger Than the Whole Sky" "Would've, Could've, Should've" "You're Losing Me"

The Tortured Poets
Department	

    "Fortnight" "The Tortured Poets Department "My Boy Only Breaks His Favorite Toys" "Down Bad" "So Long, London" "But Daddy I Love Him" "Fresh Out the Slammer" "Florida!!!" "Guilty as Sin?" "Who's Afraid of Little Old Me?" "Loml" "I Can Do It with a Broken Heart" "The Smallest Man Who Ever Lived" "The Alchemy" "Clara Bow" "The Black Dog" "So High School" "Thank You Aimee"

Soundtrack songs	

    "Crazier" "Today Was a Fairytale" "Safe & Sound" "Eyes Open" "Sweeter than Fiction" "I Don't Wanna Live Forever" "Beautiful Ghosts" "Only the Young" "Carolina"

Featured songs	

    "Two Is Better Than One" "Half of My Heart" "Both of Us" "Babe" "Gasoline" "Renegade" "The Joker and the Queen" "The Alcott" "Us"

Other songs	

    "Best Days of Your Life" "Ronan" "Highway Don't Care" "Better Man" "Christmas Tree Farm" "All of the Girls You Loved Before"

    Category

Authority control databases Edit this at Wikidata	

    MusicBrainz work

Categories:

    2012 songsTaylor Swift songsSongs written by Taylor SwiftSongs written by Liz Rose2010s balladsAmerican soft rock songsRock balladsCountry balladsSongs about heartacheSong recordings produced by Taylor SwiftSong recordings produced by Nathan Chapman (record producer)Song recordings produced by Jack AntonoffSong recordings produced by Chris RoweSongs containing the I–V-vi-IV progressionBillboard Hot 100 number-one singlesBillboard Global 200 number-one singlesBillboard Global Excl. U.S. number-one singlesCanadian Hot 100 number-one singlesIrish Singles Chart number-one singlesNumber-one singles in AustraliaNumber-one singles in MalaysiaNumber-one singles in New ZealandNumber-one singles in Singapore2021 songs2020s balladsBreakup songsAmerican country music songs

    This page was last edited on 10 January 2025, at 14:52 (UTC).
    Text is available under the Creative Commons Attribution-ShareAlike 4.0 License; additional terms may apply. By using this site, you agree to the Terms of Use and Privacy Policy. Wikipedia® is a registered trademark of the Wikimedia Foundation, Inc., a non-profit organization.

    Privacy policy
    About Wikipedia
    Disclaimers
    Contact Wikipedia
    Code of Conduct
    Developers
    Statistics
    Cookie statement
    Mobile view

    Wikimedia Foundation
    Powered by MediaWiki



Wikipedia The Free Encyclopedia

    Donate
    Create account
    Log in

Contents
(Top)
Background and production

Music and lyrics

Release and commercial performance

Critical reception

Music video

    Development and synopsis
    Release and reception

Live performances

Ryan Adams cover

Credits and personnel

Charts

    Weekly charts
    Year-end charts

Certifications

Release history

"Wildest Dreams (Taylor's Version)"

    Production and reception
    Credits and personnel
    Charts
    Certifications

See also

References

        Sources

Wildest Dreams

    Article
    Talk

    Read
    Edit
    View history

Tools

Appearance
Text

    Small
    Standard
    Large

Width

    Standard
    Wide

Color (beta)

    Automatic
    Light
    Dark

Featured article
From Wikipedia, the free encyclopedia
This article is about the Taylor Swift song. For other uses, see Wildest Dreams (disambiguation).
"Wildest Dreams"
Cover artwork of "Wildest Dreams", a black and white photo of Swift sitting
Single by Taylor Swift
from the album 1989
Released	August 31, 2015
Studio	

    MXM (Stockholm)
    Conway (Los Angeles)

Genre	

    Synth-pop dream pop electropop

Length	3:40
Label	Big Machine
Songwriter(s)	

    Taylor Swift Max Martin Shellback

Producer(s)	

    Max Martin Shellback

Taylor Swift singles chronology
"Bad Blood"
(2015) 	"Wildest Dreams"
(2015) 	"Out of the Woods"
(2016)
Music video
"Wildest Dreams" on YouTube

"Wildest Dreams" is a song by the American singer-songwriter Taylor Swift. It is the fifth single from her fifth studio album, 1989 (2014). Swift wrote the song with its producers Max Martin and Shellback. "Wildest Dreams" has an atmospheric, balladic production incorporating programmed drums, Mellotron–generated and live strings, and synthesizers; the rhythm interpolates Swift's heartbeat. Critics described it as synth-pop, dream pop, and electropop. The lyrics feature Swift pleading with a lover to remember her even after their relationship ends. Big Machine Records in partnership with Republic Records released "Wildest Dreams" to radio on August 31, 2015.

When the song was first released, some critics found the production and Swift's vocals alluring but others found the track derivative, comparing it to the music of Lana Del Rey. Retrospectively, critics have described "Wildest Dreams" as one of Swift's most memorable songs. The single peaked within the top five on charts of Australia, Canada, Poland, and South Africa. It was certified diamond in Brazil, nine-times platinum in Australia, and double platinum in Portugal and the United Kingdom. In the United States, "Wildest Dreams" peaked at number five and became 1989's fifth consecutive top-ten single on the Billboard Hot 100; it peaked atop three of Billboard's airplay charts. The Recording Industry Association of America (RIAA) certified the track four-times platinum.

Joseph Kahn directed the music video for "Wildest Dreams". Set in Africa in the 1950s, it depicts Swift as a classical Hollywood actress who falls in love with her co-star but ends the fling upon completion of their film project. Media publications praised the production as cinematic but accused the video of glorifying colonialism, a claim that Kahn dismissed. Swift included "Wildest Dreams" in the set lists for two of her world tours, the 1989 World Tour (2015) and the Eras Tour (2023–2024). Following the dispute regarding the ownership of Swift's master recordings in 2019 and the viral popularity of "Wildest Dreams" on the social media site TikTok in 2021, Swift released the re-recorded version "Wildest Dreams (Taylor's Version)".
Background and production

Taylor Swift had identified as a country musician until her fourth studio album, Red, which was released on October 22, 2012.[1] Red's eclectic pop and rock styles beyond the country stylings of Swift's past albums led to critics questioning her country-music identity.[2][3] Swift began writing songs for her fifth studio album in mid-2013 while touring.[4] She was inspired by 1980s synth-pop to create her fifth studio album, 1989, which she described as her first "official pop album" and named after her birth year.[5][6] The album makes extensive use of synthesizers, programmed drum machines, and electronic and dance stylings, a stark contrast to the acoustic arrangements of her country–styled albums.[7][8]

Swift and Max Martin served as executive producers of 1989.[9] On the album's standard edition, Martin and his frequent collaborator Shellback produced 7 out of 13 songs, including "Wildest Dreams".[10] Swift wrote "Wildest Dreams" with Martin and Shellback, who both produced and programmed the song and played the keyboards. Martin played the piano, and Shellback played the electric guitar and percussion.[10] Mattias Bylund joined the production of "Wildest Dreams" after Martin played the track to him; Bylund played and arranged the strings, which he recorded and edited at his home studio in Tuve, Sweden.[9] Michael Ilbert and Sam Holland, assisted by Cory Bice, recorded the track at MXM Studios in Stockholm and Conway Recording Studios in Los Angeles. It was mixed by Serban Ghenea at MixStar Studios in Virginia Beach and mastered by Tom Coyne at Sterling Sound in New York.[10]
Music and lyrics
"Wildest Dreams"
Duration: 21 seconds.0:21
"Wildest Dreams" has a balladic production that incorporates synths, Mellotron–generated and live strings, and it interpolates Swift's heartbeat in its rhythm. Swift sings with breathy vocals.
Problems playing this file? See media help.

"Wildest Dreams" is a power ballad that interpolates Swift's heartbeat in its rhythm.[10][11] It incorporates programmed drums, pulsing synths, and staccato strings generated with a Mellotron.[9][12][13] In the chorus, the melody is accentuated by live strings with what Bylund described as "Coldplay-type rhythm chords".[9] Swift sings with breathy vocals.[12][14][15] According to Jon Caramanica from The New York Times, she sings "drowsily" in the verses and "skips up an octave" in the bridge.[16] Jem Aswad of Billboard said that she "[flits] between a fluttery soprano and deadpan alto".[17] Music critics characterized the genre as synth-pop,[18][19] dream pop,[11] and electropop,[20] with elements of chillwave.[21] Although the synths and drums were a stark contrast to Swift's earlier music, the musicologist James E. Perone said that the composition retained some elements from her previous country songs: the "heavy use" of the pentatonic scale in the melody and the move between major and minor chords in the chorus.[13]

In the lyrics, Swift's character tells a lover to remember her after their relationship ends while still being in love with him.[15][22] The first verse is about lust: "He's so tall, and handsome as hell/ He's so bad, but he does it so well/ I can see the end as it begins."[23][24] She expresses her desire to live on in the lover's memory as a woman with red lips, "standing in a nice dress, staring at the sunset".[25][26] She cautions the lover that she will haunt him: "Say you'll see me again even if it's just in your wildest dreams."[27] The bridge is set in double time and sees Swift's character affirming, "You see me in hindsight/ Tangled up with you all night/ Burnin' it down."[28][29]

Critics have described the sound as sultry, sensual, and dramatic, comparing the production and the theme of failed romance to the music of the singer-songwriter Lana Del Rey.[14][26][30][31][32] The Guardian's Alexis Petridis felt that the song abandoned Swift's previous "persona of the pathetic female appendage snivelling over her bad-boy boyfriend" and instead portrayed the man as her victim.[22] Slate's Forrest Wickman thought that Swift's character was a "sort of [...] femme fatale".[27] Robert Leedham of Drowned in Sound wrote that the lyrics portrayed her arrogance and confidence to "[move] onto better things", contrasting with the victim mentality on her past songs.[32]
Release and commercial performance

Big Machine Records released 1989 on October 27, 2014; "Wildest Dreams" is number nine on the standard edition's track listing.[10][33] The song debuted at number 76 on the US Billboard Hot 100 in November 2014.[34] On August 5, 2015, Swift shared on Twitter that "Wildest Dreams" would be the fifth single from 1989, following four Hot 100 top-10 singles: "Shake It Off", "Blank Space", "Style", and "Bad Blood".[35] In the United States, Big Machine and Republic Records released "Wildest Dreams" to hot adult contemporary radio on August 31,[36] and contemporary hit radio on September 1, 2015.[37] Big Machine released a remix by R3hab for download via the iTunes Store on October 15,[38] and Universal Music released the original version to Italian radio on October 30.[39]

On the Billboard Hot 100, "Wildest Dreams" re-entered at number 15 on the chart dated September 19, 2015, after its single release.[40] It reached number 10 on October 10, 2015, and became 1989's fifth consecutive top-10 single.[41] In the Billboard issue dated November 7, 2015, the single peaked at number five on the Hot 100 and became 1989's fifth consecutive number-one song on two Billboard's airplay charts: Pop Songs and Adult Pop Songs; 1989 tied with Katy Perry's Teenage Dream (2010) to become the album with the most Adult Pop Songs number ones.[42] On Billboard's Dance/Mix Show Airplay chart, supported by the R3hab remix, "Wildest Dreams" was Swift's first number one and made her the first female artist to have five top-10 songs in a calendar year.[43] "Wildest Dreams" was certified four-times platinum by the Recording Industry Association of America (RIAA) and had sold two million digital copies in the United States by November 2017.[44]

"Wildest Dreams" reached the top 10 on the singles charts of Canada (4),[45] South Africa (5),[46] Venezuela (6),[47] Iceland (8),[48] New Zealand (8),[49] Slovakia (8),[50] and Scotland (9).[51] It received platinum or higher certifications in Brazil (diamond),[52] Portugal (double platinum),[53] and the United Kingdom (double platinum).[54] It received platinum certifications in Austria,[55] Canada,[56] Denmark,[57] Italy,[58] and Spain.[59] The track also received gold certifications in New Zealand,[60] Germany,[61] and Norway.[62] In Australia, the single peaked at number three on the ARIA Singles Chart[63] and was certified nine-times platinum by the Australian Recording Industry Association (ARIA).[64]
Critical reception

When it was first released, "Wildest Dreams" received mixed reviews from music critics. Petridis found it "hugely cheering" that Swift employed a new perspective in her songwriting.[22] Caramanica said that the song had the "most pronounced vocal tweak" on 1989, demonstrating Swift's new ways of expressing herself in music.[16] The Arizona Republic's Ed Masley found the track "haunting" and Swift's vocals "seductive".[20] Sam Lansky of Time described the production as "lush" and full of "cinematic grandeur".[23] Writing for Hot Press, Paul Nolan picked it as the album's best track for its combination of chillwave and "sweeping, singalong choruses".[21] The song helped Swift win the Songwriter of the Year Award at the 2016 BMI Pop Awards[65] and was recognized at the 2017 ASCAP Awards.[66]

Other reviews opined that the track was influenced by Lana Del Rey to an extent that it erased Swift's authenticity.[67] Aswad said that it was "hard to tell if the song is homage or parody",[17] and Wickman and Mikael Wood of the Los Angeles Times opined that Swift's songwriting lost its distinctive quality.[12][27] Shane Kimberline from MusicOMH and Lindsay Zoladz from Vulture deemed "Wildest Dreams" one of the album's weakest tracks and took issue with the Del Rey resemblance in Swift's vocals and lyrics.[14][68] Slant Magazine's Annie Galvin said Swift's vocals complemented the narrative lyrics but described the song as a "misguided imitation" of Del Rey with a predictable storyline.[24] In The Atlantic, Kevin O'Keeffe argued that the Del Rey comparisons were "unfair", and Emma Green praised the storytelling lyrics and contended that they were "unabashed, all-consuming, earnest nostalgia, anticipating", which she deemed distinct from Del Rey's "performative, cool-girl nostalgia".[15]

Retrospectively, Rob Sheffield of Rolling Stone wrote in 2019 that the song "sounds stronger and stronger over the years".[69] NME's Hannah Mylrae called it a "beauty",[18] and Nate Jones from Vulture considered it one of Swift's 10 best songs and specifically lauded the "invigorating double-time bridge".[28] The bridge was ranked 66th on Billboard's 2021 list of the 100 Greatest Song Bridges of the 21st Century.[29] Alex Hudson and Megan LaPierre of Exclaim! included "Wildest Dreams" in their list of the best 20 songs by Swift, saying that she "totally nails" the Del Rey resemblance.[70] Jane Song from Paste lauded the "dark Lana Del Rey-esque pop" production and opined that the lyrics about memory made the song have "more staying power than you'd expect".[71] Petridis ranked it 18th out of 44 singles Swift had released by April 2019, and he said that the song employed a Del Rey-inspired songwriting trope with a "smart, pleasing twist".[72]
Music video
Development and synopsis

Joseph Kahn directed the music video for "Wildest Dreams",[73] the third time he directed a music video for a 1989 single after "Blank Space" and "Bad Blood".[74] Filming primarily took place in Botswana and South Africa. Inspired by The Secret Conversations (2013), a memoir of the actress Ava Gardner,[75][76] Swift conceived the premise for the video as an illicit love affair between two actors in an isolated place within Africa, because they could only interact with each other without other means of communication.[77] Kahn took inspiration from romantic films set in Africa, such as The African Queen (1951), Out of Africa (1985), and The English Patient (1996).[78]

The video's narrative focuses on an affair between a classical Hollywood actress (Swift) and her male co-star (Scott Eastwood) who shoot a film in 1950s Africa.[79][80] Kahn compared the affair to the romance between Elizabeth Taylor and Richard Burton.[78] The pair gets involved romantically off-screen, as the video features shots of wildlife such as giraffes, zebras, and lions in a broad savanna.[81] The affair turns sour after a fight on set.[80] As the romance ends, the pair is seen shooting in front of a savanna backdrop in a California studio.[79] At the film's premiere, Swift's character sees her co-star with his wife. During the screening, Swift's character flees the theater and gets into a waiting limousine, as the co-star runs into the street and watches her leave.[82]
Release and reception

The video premiered on television during the pre-show of the 2015 MTV Video Music Awards on August 31.[83] Swift donated all of the proceeds from the video to the African Parks Foundation of America for wild animal conservation causes.[77] Rolling Stone's Brittany Spanos commented that Swift and Eastwood channeled "retro Hollywood glamour",[84] and Billboard's Natalie Weiner deemed Elizabeth Taylor an influence on Swift's fashion in the video.[85] ABC News described the video as visually powerful,[86] and Wickman found the production cinematic and the narrative "a lot more engaging" than the music video for "Style".[87] Mike Wass of Idolator said that although Swift and Eastwood did not have a strong "chemistry", the African scenery and narrative "all [hang] together rather nicely".[88] The video was nominated for Best Fresh Video at the 2016 MTV Italian Music Awards.[89]

Many online blogs and publications contended that the video glorified "white colonialism" by featuring a white cast in Africa.[90] Critics opined that it portrayed a romanticized nostalgia for colonial Africa held by white people and neglected the struggles of the African peoples during the European colonization.[91][92][93] The African studies professor Matthew Carotenuto wrote that the storyline depicted "pith-helmet-and-khaki-clad men as civilizing heroes and the women who joined them roughing it in tents wearing lingerie".[94] In the book Mistaking Africa, the history and political science authors Curtis Keim and Carolyn Somerville wrote that "Wildest Dreams" reinforced the stereotypes associated with Africa and "the mistaken perception held by many Americans that large game are found everywhere in Africa and that all parts of Africa are identical".[95] Kahn defended the video and said that featuring a black cast would be historically inaccurate for the 1950s settings.[96] Lauretta Charlton of Vulture felt that the accusations were overblown: although she acknowledged that the video's depiction of Africa was problematic, she regarded it as "antiquated" and recommended the audience to focus on the "modern-day colonialism of Africa" that demanded urgent attention.[97]

Some journalists and academics analyzed the video in the context of Swift's celebrity and the historical Hollywood depictions of Africa. Carotenuto opined that Swift was part of a "Lion King generation", which led her to think of Africa as "nothing more than a rich tapestry of flora and fauna, with actual Africans fading onto the periphery", an idea that had been propagated by Hollywood films and popular American culture.[94] The Atlantic's Spencer Kornhaber wrote that her generation was when "certain symbols of white dominance [...] have been glorified". For Kornhaber, "Wildest Dreams" was in line with Swift's artistic vision of "a powerful but vague nostalgia, defined less by time period than by particular strains of influence that just happen to be affiliated with a certain skin color".[25] Kornhaber and Tshepo Mokoena from The Guardian argued that the criticism was not meant to portray Swift as racist. The former contended that it was a "lesson" for Swift about "how nostalgia can be inherently political";[25] the latter said that the video was a "clumsy move, but not one that merits outrage", but the criticism blemished Swift's "America's sweetheart" reputation.[93]
Live performances
Swift in a rhinestoned blue dress onstage
Swift performed "Wildest Dreams" on the Eras Tour.

On the 1989 World Tour (2015), Swift performed the song as part of a mashup with "Enchanted", from her 2010 album Speak Now.[98] Playing a sparkling grand piano, she first sang "Wildest Dreams" and, after the second chorus, proceeded with "Enchanted". The rendition built up with accompanying synths and backing vocals.[99][100] She finished the mashup by changing costumes from a sparkling tulle skirt to a bodysuit for the next number.[100] The Ringer's Nora Princiotti in March 2023 deemed it Swift's best live performance, praising it as an "epic five-and-a-half-minute medley [that] is fundamentally simple".[100]

"Wildest Dreams" was included in Swift's other concerts. On September 30, 2015, she performed a stripped-down rendition on an electric guitar as part of the "Taylor Swift Experience" exhibition at the Grammy Museum at L.A. Live.[101] At a private concert for 100 fans in Hamilton Island, Australia, as part of Nova's "Red Room" series, Swift "Wildest Dreams" on an acoustic guitar.[102] Swift included the "Wildest Dreams"/"Enchanted" mashup in the set lists of two concerts: at the United States Grand Prix in Austin on October 22, 2016,[103] and at the Super Saturday Night, a pre-Super Bowl event, on February 4, 2017.[104]

Swift performed "Wildest Dreams" as a "surprise song" outside the regular set list twice on her Reputation Stadium Tour in 2018: at the first show in Santa Clara, California, on May 11, and at the second show in Tokyo, Japan, on November 21.[105] At the Philadelphia concert of the Reputation Stadium Tour on July 14, she sang "Wildest Dreams" a cappella after a stage device malfunctioned.[106] On the Eras Tour (2023–2024), a tour that Swift described as a tribute to all of her albums, she performed the song as part of the 1989 act as the screen projected scenes of a couple in bed.[107]
Ryan Adams cover

The singer-songwriter Ryan Adams released his track-by-track cover album of 1989 on September 21, 2015.[108] Adams said that Swift's 1989 helped him cope with emotional hardships and that he wanted to sing the songs from his perspective "like it was Bruce Springsteen's Nebraska".[109] Before the album's release, Adams previewed his cover of "Wildest Dreams" online in August.[110] He switches and adjusts pronouns in some places; the lyric "Standing in a nice dress" becomes "Standing in your nice dress".[111] His version combines country rock, alternative country, and jangle pop.[112][113][114] It uses acoustic instruments of live drums and guitar strums.[115][116][117]

Adams's "Wildest Dreams" peaked at number 40 on Billboard's Hot Rock & Alternative Songs chart.[118] Kornharber found the cover "undeniably lovely",[119] Jeremy Winograd of Slant Magazine deemed it a tasteful incorporation of 1980s rock,[113] and Marc Burrows of Drowned in Sound preferred Adams's cover to Swift's version.[112] Sarah Murphy in Exclaim! labeled the cover "an equally impressive feat" that could resonate with Swift's fans who lamented her departure from country music.[114] In The Guardian, Michael Cragg said that there were no substantial additions in Adams's cover, which he described as a "fairly rudimentary strumalong",[117] and Rachel Aroesti found it "comical" that it failed to match the original.[120] Caramanica said that the lyrical alterations brought "no real effect".[121]
Credits and personnel

Credits are adapted from liner notes of 1989.[10]

    Taylor Swift – vocals, writer, heart sounds
    Max Martin – producer, writer, keyboard, piano, programming
    Shellback – producer, writer, acoustic guitar, electric guitar, keyboard, percussion, programming
    Mattias Bylund – string arrangements, recording, and editing
    Michael Ilbert – recording
    Sam Holland – recording
    Cory Bice – assistant recording
    Serban Ghenea – mixing
    John Hanes – engineered for mix
    Tom Coyne – mastering

Charts
Weekly charts
2015–2016 weekly chart performance for "Wildest Dreams" Chart (2015–2016) 	Peak
position
Australia (ARIA)[63] 	3
Austria (Ö3 Austria Top 40)[122] 	21
Belgium (Ultratop 50 Flanders)[123] 	33
Belgium (Ultratip Bubbling Under Wallonia)[124] 	24
Canada (Canadian Hot 100)[45] 	4
Canada AC (Billboard)[125] 	2
Canada CHR/Top 40 (Billboard)[126] 	1
Canada Hot AC (Billboard)[127] 	1
Czech Republic (Rádio – Top 100)[128] 	25
Finland Airplay (Radiosoittolista)[129] 	7
Finland Download (Latauslista)[130] 	15
France (SNEP)[131] 	122
Greece Digital Songs (Billboard)[132] 	3
Hungary (Single Top 40)[133] 	35
Iceland (RÚV)[48] 	8
Ireland (IRMA)[134] 	39
Lebanon (Lebanese Top 20)[135] 	14
Mexico (Billboard Mexico Airplay)[136] 	46
Mexico Anglo (Monitor Latino)[137] 	16
Netherlands (Dutch Top 40 Tipparade)[138] 	12
New Zealand (Recorded Music NZ)[49] 	8
Poland (Polish Airplay Top 100)[139] 	3
Scotland (OCC)[51] 	9
Slovakia (Rádio Top 100)[50] 	8
Slovenia (SloTop50)[140] 	14
South Africa (EMA)[46] 	5
UK Singles (OCC)[141] 	40
US Billboard Hot 100[142] 	5
US Adult Contemporary (Billboard)[143] 	2
US Adult Pop Airplay (Billboard)[144] 	1
US Dance/Mix Show Airplay (Billboard)[145] 	1
US Pop Airplay (Billboard)[146] 	1
US Rhythmic (Billboard)[147] 	25
Venezuela (Record Report)[47] 	6
2021 weekly chart performance for "Wildest Dreams" Chart (2021) 	Peak
position
Czech Republic (Singles Digitál Top 100)[148] 	28
Germany (GfK)[149] 	51
Global 200 (Billboard)[150] 	170
Portugal (AFP)[151] 	57
Slovakia (Singles Digitál Top 100)[152] 	47
Sweden (Sverigetopplistan)[153] 	53
Switzerland (Schweizer Hitparade)[154] 	53
	
Year-end charts
2015 year-end charts for "Wildest Dreams" Chart (2015) 	Position
Australia (ARIA)[155] 	41
Canada (Canadian Hot 100)[156] 	39
US Billboard Hot 100[157] 	57
US Adult Contemporary (Billboard)[158] 	27
US Adult Top 40 (Billboard)[159] 	28
US Mainstream Top 40 (Billboard)[160] 	30
2016 year-end charts for "Wildest Dreams" Chart (2016) 	Position
Canada (Canadian Hot 100)[161] 	70
US Billboard Hot 100[162] 	79
US Adult Contemporary (Billboard)[163] 	3
US Adult Top 40 (Billboard)[164] 	28

Certifications
Certifications for "Wildest Dreams" Region 	Certification 	Certified units/sales
Australia (ARIA)[64] 	9× Platinum 	630,000‡
Austria (IFPI Austria)[55] 	Platinum 	30,000‡
Brazil (Pro-Música Brasil)[52] 	Diamond 	250,000‡
Canada (Music Canada)[56] 	Platinum 	80,000*
Denmark (IFPI Danmark)[57] 	Platinum 	90,000‡
Germany (BVMI)[61] 	Gold 	200,000‡
Italy (FIMI)[58] 	Platinum 	100,000‡
New Zealand (RMNZ)[165] 	3× Platinum 	90,000‡
Norway (IFPI Norway)[62] 	Gold 	30,000‡
Portugal (AFP)[53] 	2× Platinum 	20,000‡
Spain (PROMUSICAE)[59] 	Platinum 	60,000‡
United Kingdom (BPI)[54] 	2× Platinum 	1,200,000‡
United States (RIAA)[166] 	4× Platinum 	4,000,000‡

* Sales figures based on certification alone.
‡ Sales+streaming figures based on certification alone.
Release history
Release dates and formats for "Wildest Dreams" Region 	Date 	Format(s) 	Label(s) 	Ref.
United States 	August 31, 2015 	Adult contemporary radio 	

    Big MachineRepublic

	[36]
September 1, 2015 	Contemporary hit radio 	[37]
Various 	October 15, 2015 	

    Digital downloadstreaming

(R3hab remix) 	Big Machine 	[38]
Italy 	October 30, 2015 	Radio airplay 	Universal 	[39]
"Wildest Dreams (Taylor's Version)"
"Wildest Dreams (Taylor's Version)"
Taylor Swift standing under the sunlight, looking to her left, wearing sunglasses and a horizontally stripped T-shirt.
Promotional single by Taylor Swift
from the album 1989 (Taylor's Version)
Released	September 17, 2021
Studio	Kitty Committee (London)
Genre	Synth-pop
Length	3:40
Label	Republic
Songwriter(s)	

    Taylor Swift Max Martin Shellback

Producer(s)	

    Taylor Swift Shellback Christopher Rowe

Audio video
"Wildest Dreams (Taylor's Version)" on YouTube

Swift departed from Big Machine and signed a new contract with Republic Records in 2018. She began re-recording her first six studio albums in November 2020.[167] The decision followed a 2019 dispute between Swift and the talent manager Scooter Braun, who acquired Big Machine Records, over the masters of Swift's albums that the label had released.[168][169] By re-recording the albums, Swift had full ownership of the new masters, which enabled her to encourage licensing of her re-recorded songs for commercial use in hopes of substituting the Big Machine-owned masters.[170] She denoted the re-recordings with a "Taylor's Version" subtitle.[171]

The re-recording of "Wildest Dreams" is titled "Wildest Dreams (Taylor's Version)". Its snippets were featured in the March–May trailers for the 2021 animated film Spirit Untamed by DreamWorks Animation.[172][173][174] On September 17, 2021, Swift released "Wildest Dreams (Taylor's Version)" onto digital and streaming platforms. The release followed the viral success of the original song on the video-sharing platform TikTok, which lead to an increase in streams.[175][176] "Wildest Dreams (Taylor's Version)" is included as part of 1989 (Taylor's Version), the re-recorded version of 1989, which was released on October 27, 2023.[177]
Production and reception

"Wildest Dreams (Taylor's Version)" is a synth-pop song that replicates the original's production.[19][176][178] Swift produced the song with Shellback and Christopher Rowe, a Nashville-based vocal engineer who had produced her re-recorded album Fearless (Taylor's Version).[179] Although Martin did not return as producer, the musicians were those from Swift's backing band during the 1989 sessions.[19] Aroesti remarked that the re-recorded version was almost identical to the original but was "sometimes bassier".[180] Robin Murray of Clash said that it contained "subtle stylist[ic] shifts",[181] and Stereogum's Tom Breihan found it more "muted".[19] Mary Siroky of Consequence appreciated that the production took "great care to capture the sound of the original, right down to a riff in the second chorus".[178] Murray and Siroky praised Swift's vocals as having improved.[178][181]

Within four hours, "Wildest Dreams (Taylor's Version)" amassed over two million streams on Spotify, surpassing the original version's biggest single-day streaming tally on the platform.[182] In the United States, it debuted at number 37 on the Billboard Hot 100 for the week ending September 23, 2021,[183] with 13,400 downloads and 8.7 million streams.[184] In both Ireland and the United Kingdom, "Wildest Dreams (Taylor's Version)" surpassed the peak positions of the original version (15–39 and 25–40).[185][186] After 1989 (Taylor's Version) was released, "Wildest Dreams (Taylor's Version)" peaked at number 19 on the Billboard Global 200[150] and re-entered and peaked at number 19 on the Hot 100 chart dated November 11, 2023.[142][187] The song reached the top 10 in Malaysia (10) and Singapore (5).[188][189] It peaked in the top 40 in Australia (14),[190] the Philippines (15),[191] Canada (18),[45] Hungary (29),[192] and New Zealand (30).[193] It was certified double platinum in Australia,[64] gold in New Zealand and Greece,[194][195] and silver in the United Kingdom.[196]
Credits and personnel

    Taylor Swift – lead vocals, songwriting, production
    Christopher Rowe – production, vocal engineering
    Shellback – songwriting, production
    Max Martin – songwriting
    Mattias Bylund – record engineering, editing, strings arrangement, synthesizer
    Max Bernstein – guitar, synthesizer, synthesizer programming
    Mike Meadows – synthesizer, synthesizer programming
    Dan Burns – synthesizer programming
    Matt Billingslea – drums, percussion
    Amos Heller – bass
    Paul Sidoti – guitar
    Mattias Johansson – violin
    David Bukovinszky – cello
    Serban Ghenea – mixing
    John Hanes – engineering
    Randy Merrill – master engineering

Charts
Weekly chart performance for "Wildest Dreams (Taylor's Version)" Chart (2021–2023) 	Peak
position
Australia (ARIA)[190] 	14
Canada (Canadian Hot 100)[45] 	18
Canada AC (Billboard)[125] 	33
Euro Digital Song Sales (Billboard)[197] 	10
France (SNEP)[198] 	182
Global 200 (Billboard)[150] 	19
Greece International (IFPI)[195] 	39
Hungary (Single Top 40)[192] 	29
Ireland (IRMA)[199] 	15
Lithuania (AGATA)[200] 	88
Malaysia (RIM)[188] 	10
Netherlands (Single Top 100)[201] 	69
New Zealand (Recorded Music NZ)[193] 	30
Philippines (Billboard)[191] 	15
Singapore (RIAS)[189] 	5
South Africa (RISA)[202] 	63
Sweden (Sverigetopplistan)[203] 	95
UK Singles (OCC)[204] 	25
US Billboard Hot 100[142] 	19
US Adult Contemporary (Billboard)[143] 	23
Vietnam (Vietnam Hot 100)[205] 	80
Certifications
Certifications for "Wildest Dreams (Taylor's Version)" Region 	Certification 	Certified units/sales
Australia (ARIA)[206] 	2× Platinum 	140,000‡
Brazil (Pro-Música Brasil)[52] 	2× Platinum 	80,000‡
New Zealand (RMNZ)[194] 	Gold 	15,000‡
Poland (ZPAV)[207] 	Gold 	25,000‡
United Kingdom (BPI)[196] 	Gold 	400,000‡
Streaming
Greece (IFPI Greece)[195] 	Gold 	1,000,000†

‡ Sales+streaming figures based on certification alone.
† Streaming-only figures based on certification alone.
See also

    List of Billboard Adult Top 40 number-one songs of the 2010s
    List of Billboard Mainstream Top 40 number-one songs of 2015
    List of Billboard Hot 100 top-ten singles in 2015

References

Caulfield, Keith (October 30, 2012). "Taylor Swift's Red Sells 1.21 Million; Biggest Sales Week for an Album Since 2002". Billboard. Archived from the original on February 1, 2013. Retrieved February 4, 2019.
McNutt 2020, p. 77.
Light, Alan (December 5, 2014). "Billboard Woman of the Year Taylor Swift on Writing Her Own Rules, Not Becoming a Cliche and the Hurdle of Going Pop". Billboard. Archived from the original on December 26, 2014. Retrieved February 27, 2019.
Talbott, Chris (October 13, 2013). "Taylor Swift Talks Next Album, CMAs and Ed Sheeran". Associated Press. Archived from the original on October 26, 2013. Retrieved October 26, 2013.
Eells, Josh (September 16, 2014). "Taylor Swift Reveals Five Things to Expect on 1989". Rolling Stone. Archived from the original on November 16, 2018. Retrieved November 16, 2018.
Sisario, Ben (November 5, 2014). "Sales of Taylor Swift's 1989 Intensify Streaming Debate". The New York Times. Archived from the original on November 11, 2014. Retrieved February 27, 2019.
Pettifer, Amy (November 27, 2014). "Reviews: Taylor Swift, 1989". The Quietus. Retrieved June 21, 2023.
Perone 2017, p. 55–56.
Zollo, Paul (February 13, 2016). "The Oral History of Taylor Swift's 1989". The Recording Academy. Archived from the original on April 4, 2016. Retrieved March 23, 2016 – via Cuepoint.
1989 (Compact disc liner notes). Taylor Swift. Big Machine Records. 2014. BMRBD0500A.
"Taylor Swift – 'Wildest Dreams' (video) (Singles Going Steady)". PopMatters. September 1, 2015. Archived from the original on December 28, 2017. Retrieved December 27, 2017.
Wood, Mikael (October 27, 2014). "Taylor Swift Smooths Out the Wrinkles on Sleek 1989". Los Angeles Times. Archived from the original on November 15, 2014. Retrieved November 15, 2020.
Perone 2017, p. 62.
Kimberlin, Shane (November 3, 2014). "Taylor Swift – 1989 | Album Review". MusicOMH. Archived from the original on November 5, 2014. Retrieved February 5, 2019.
Cruz, Lenika; Beck, Julie; Green, Emma; O'Keeffe, Kevin (October 25, 2014). "Taylor Swift's 1989: First Impressions". The Atlantic. Archived from the original on June 24, 2023. Retrieved December 28, 2023.
Caramanica, Jon (October 23, 2014). "A Farewell to Twang". The New York Times. Archived from the original on November 11, 2014. Retrieved August 30, 2015.
Aswad, Jem (October 24, 2014). "Album Review: Taylor Swift's Pop Curveball Pays Off With 1989". Billboard. Archived from the original on February 8, 2017. Retrieved August 30, 2015.
Mylrea, Hannah (September 8, 2020). "Every Taylor Swift Song Ranked In Order of Greatness". NME. Archived from the original on September 8, 2020. Retrieved September 8, 2020.
Breihan, Tom (September 17, 2021). "Taylor Swift Shares Her Re-Recorded Version of 'Wildest Dreams': Listen". Stereogum. Archived from the original on May 6, 2022. Retrieved May 6, 2022.
Masley, Ed (August 12, 2015). "30 Best Taylor Swift Singles Ever (So Far)". The Arizona Republic. Retrieved July 30, 2020.
Nolan, Paul (November 13, 2014). "Taylor Swift 1989 – Album Review". Hot Press. Retrieved December 29, 2023.
Petridis, Alexis (October 24, 2014). "Taylor Swift: 1989 Review – Leagues Ahead of the Teen-Pop Competition". The Guardian. Archived from the original on November 1, 2014. Retrieved February 4, 2019.
Lansky, Sam (October 23, 2014). "Review: 1989 Marks a Paradigm Swift". Time. Archived from the original on November 1, 2014. Retrieved December 28, 2023.
Galvin, Annie (October 27, 2014). "Review: Taylor Swift, 1989". Slant Magazine. Archived from the original on March 5, 2019. Retrieved October 28, 2014.
Kornhaber, Spencer (September 2, 2015). "The Backlash to Taylor Swift's 'Wildest Dreams' Shows the Danger of Nostalgia". The Atlantic. Archived from the original on December 28, 2023. Retrieved December 28, 2023.
Cliff, Aimee (October 30, 2014). "1989". Fact. Archived from the original on June 18, 2020. Retrieved December 28, 2023.
Wickman, Forrest (October 24, 2014). "Taylor Swift's 1989: A Track-by-Track Breakdown". Slate. Archived from the original on February 14, 2019. Retrieved November 20, 2020.
Jones, Nate (August 13, 2020). "Taylor Swift Songs, Ranked from Worst to Best". Vulture. Archived from the original on December 7, 2023. Retrieved November 20, 2020.
"The 100 Greatest Song Bridges of the 21st Century: Staff Picks". Billboard. May 13, 2021. Archived from the original on February 28, 2023. Retrieved December 28, 2023.
Eakin, Marah (October 28, 2014). "With 1989, Taylor Swift Finally Grows Up". The A.V. Club. Archived from the original on May 20, 2017. Retrieved August 30, 2015.
Zoladz, Lindsay (October 24, 2014). "Did Taylor Swift Rip Off Lorde and Lana Del Rey?". Vulture. Archived from the original on December 28, 2023. Retrieved December 28, 2023.
Leedham, Robert (October 27, 2014). "Album Review: Taylor Swift - 1989". Drowned in Sound. Archived from the original on February 14, 2019. Retrieved December 28, 2023.
Graff, Gary (October 24, 2014). "Taylor Swift to the Haters: 'If You're Upset That I'm Just Being Myself, I'm Going to Be Myself More'". Billboard. Archived from the original on November 2, 2023. Retrieved December 28, 2023.
Trust, Gary (November 5, 2014). "Taylor Swift's 'Shake It Off' Returns to No. 1 on Hot 100". Billboard. Archived from the original on November 7, 2014. Retrieved November 7, 2014.
Caulfield, Keith (August 5, 2015). "Taylor Swift Announces Next Single". Billboard. Archived from the original on April 15, 2021. Retrieved November 20, 2020.
"Hot/Modern/AC Future Releases". All Access Media Group. Archived from the original on August 24, 2015. Retrieved September 1, 2015.
"Top 40/M Future Releases". All Access Media Group. Archived from the original on August 24, 2015. Retrieved October 16, 2015.
"Wildest Dreams (R3hab Remix) – Single by Taylor Swift". iTunes Store. Archived from the original on October 17, 2015. Retrieved October 16, 2015.
"Taylor Swift 'Wildest Dreams'" (in Italian). Airplay Control S.R.L. Archived from the original on June 25, 2022. Retrieved June 24, 2022.
Trust, Gary (September 8, 2015). "The Weeknd Doubles Up in Hot 100's Top Three". Billboard. Archived from the original on September 10, 2015. Retrieved September 9, 2015.
Trust, Gary (September 28, 2015). "The Weeknd Holds Atop Hot 100, Taylor Swift Hits Top 10 With 'Wildest Dreams'". Billboard. Archived from the original on December 1, 2015. Retrieved December 1, 2015.
Trust, Gary (October 26, 2015). "The Weeknd Tops Hot 100; Adele No. 1 Next Week?". Billboard. Archived from the original on December 1, 2015. Retrieved December 1, 2015.
Trust, Gary (November 24, 2015). "Taylor Swift Tallies First Dance/Mix Show Airplay No. 1 With 'Wildest Dreams'". Billboard. Archived from the original on November 26, 2015. Retrieved November 24, 2015.
Trust, Gary (November 26, 2017). "Ask Billboard: Taylor Swift's Career Album & Song Sales". Billboard. Archived from the original on November 25, 2018. Retrieved November 25, 2018.
"Taylor Swift Chart History (Canadian Hot 100)". Billboard. Retrieved December 13, 2023.
"EMA Top 10 Airplay: Week Ending 2015-10-06". Entertainment Monitoring Africa. Retrieved October 7, 2015.
"Record Report – Rock General" (in Spanish). Record Report. Archived from the original on October 30, 2015. Retrieved October 31, 2015.
"Taylor Swift Chart History". RÚV. April 11, 2016. Archived from the original on September 3, 2017. Retrieved May 28, 2017.
"Taylor Swift – Wildest Dreams". Top 40 Singles. Retrieved September 4, 2015.
"ČNS IFPI" (in Slovak). Hitparáda – Radio Top 100 Oficiálna. IFPI Czech Republic. Note: insert 20161 into search. Retrieved January 12, 2016.
"Official Scottish Singles Sales Chart Top 100". Official Charts Company. Retrieved September 18, 2015.
"Brazilian single certifications – Taylor Swift – Wildest Dreams" (in Portuguese). Pro-Música Brasil. Retrieved July 24, 2024.
"Portuguese single certifications – Taylor Swift – Wildest Dreams" (PDF) (in Portuguese). Associação Fonográfica Portuguesa. Retrieved January 31, 2023.
"British single certifications – Taylor Swift – Wildest Dreams". British Phonographic Industry. Retrieved February 23, 2024.
"Austrian single certifications – Taylor Swift – Wildest Dreams" (in German). IFPI Austria. Retrieved May 29, 2024.
"Canadian single certifications – Taylor Swift – Wildest Dreams". Music Canada. Retrieved October 2, 2015.
"Danish single certifications – Taylor Swift – Wildest Dreams". IFPI Danmark. Retrieved April 24, 2024.
"Italian single certifications – Taylor Swift – Wildest Dreams" (in Italian). Federazione Industria Musicale Italiana. Retrieved September 16, 2024.
"Spanish single certifications – Taylor Swift – Wildest Dreams". El portal de Música. Productores de Música de España. Retrieved April 1, 2024.
Cite error: The named reference rmnz was invoked but never defined (see the help page).
"Gold-/Platin-Datenbank (Taylor Swift; 'Wildest Dreams')" (in German). Bundesverband Musikindustrie. Retrieved November 18, 2022.
"Norwegian single certifications – Taylor Swift – Wildest Dreams" (in Norwegian). IFPI Norway. Retrieved November 27, 2020.
"Taylor Swift – Wildest Dreams". ARIA Top 50 Singles. Retrieved September 19, 2015.
"ARIA Charts – Accreditations – 2024 Singles" (PDF). Australian Recording Industry Association. Retrieved June 14, 2024.
"BMI Honors Taylor Swift and Legendary Songwriting Duo Mann & Weil at the 64th Annual BMI Pop Awards". Broadcast Music, Inc. May 11, 2016. Archived from the original on May 27, 2016. Retrieved May 11, 2016.
"ASCAP Pop Awards 2017". American Society of Composers, Authors and Publishers. Archived from the original on June 23, 2018. Retrieved October 20, 2018.
Manning, Craig (October 31, 2014). "Taylor Swift – 1989 – Album Review". AbsolutePunk. Archived from the original on October 29, 2014. Retrieved August 30, 2015.
Zoladz, Lindsay (October 27, 2014). "Taylor Swift's 1989 Is Her Most Conservative Album Yet". Vulture. Archived from the original on January 10, 2018. Retrieved December 28, 2023.
Sheffield, Rob (December 12, 2019). "'Wildest Dreams' (2014)". Rolling Stone. Archived from the original on September 24, 2021. Retrieved November 20, 2020.
Hudson, Alex; LaPierre, Megan (October 20, 2022). "Taylor Swift's 20 Best Songs Ranked". Exclaim!. Archived from the original on December 6, 2022. Retrieved December 28, 2023.
Song, Jane (February 11, 2020). "All 158 Taylor Swift Songs, Ranked". Paste. Archived from the original on June 25, 2023. Retrieved December 28, 2023.
Petridis, Alexis (April 26, 2019). "Taylor Swift's Singles – Ranked!". The Guardian. Archived from the original on April 27, 2019. Retrieved December 28, 2023.
Dyer 2016, p. 301.
Fisher, Lauren Alexis (August 31, 2015). "Watch Taylor Swift's 'Wildest Dreams' Video". Harper's Bazaar. Retrieved January 8, 2024.
Klosterman, Chuck (October 15, 2015). "Taylor Swift on 'Bad Blood', Kanye West, and How People Interpret Her Lyrics". GQ. Archived from the original on October 18, 2015. Retrieved October 18, 2015.
Mondello, Bob (July 20, 2013). "You'll Want To Hang Up On These Secret Conversations". NPR. Archived from the original on January 12, 2021. Retrieved November 25, 2020.
Dyer 2016, p. 307.
Silver, Marc (September 2, 2015). "Director of Taylor Swift's New Video Defends His Work". NPR. Archived from the original on September 4, 2015. Retrieved September 3, 2015.
Dyer 2016, p. 308.
Linder, Emilee (August 30, 2015). "Taylor Swift's 'Wildest Dreams' Video Is Here To Make You Cry". MTV. Archived from the original on August 31, 2015. Retrieved September 1, 2015.
Dyer 2016, p. 308; Keim & Somerville 2021, p. 149.
Meynes, Carolyn (August 30, 2015). "Taylor Swift 'Wildest Dreams' Music Video: Scott Eastwood and 1989 Star Fall In Love". Music Times. Archived from the original on September 8, 2015. Retrieved December 14, 2023.
Hosken, Patrick (August 23, 2015). "Taylor Swift's 'Wildest Dreams' Will Premiere During the VMA Pre-Show". MTV News. Archived from the original on August 26, 2015. Retrieved September 4, 2015.
Spanos, Brittany (August 30, 2015). "Watch Taylor Swift's Glamorous, Retro 'Wildest Dreams' Video". Rolling Stone. Archived from the original on September 2, 2015. Retrieved September 2, 2015.
Weiner, Natalie (August 31, 2015). "Taylor Swift Debuts 'Wildest Dreams' Video at 2015 VMAs: Watch". Billboard. Archived from the original on December 14, 2023. Retrieved December 14, 2023.
"MTV VMAs: Taylor Swift Debuts 'Wildest Dreams' Video". ABC News. August 31, 2015. Archived from the original on December 14, 2023. Retrieved December 14, 2023.
Wickman, Forrest (August 30, 2015). "Watch Taylor Swift Go Classical Hollywood With the Video for 'Wildest Dreams'". Slate. Archived from the original on September 2, 2015. Retrieved September 2, 2015.
Wass, Mike (August 30, 2015). "Taylor Swift And Scott Eastwood Pay Tribute To Cinema's Greatest Love Stories In 'Wildest Dreams': Watch". Idolator. Archived from the original on September 4, 2015. Retrieved September 2, 2015.
"TRL Awards 2016" (in Italian). MTV Italy. Archived from the original on June 22, 2016. Retrieved June 19, 2016.
Khomami, Nadia (September 2, 2015). "Taylor Swift Accused of Racism In 'African Colonial Fantasy' Video". The Guardian. Archived from the original on November 8, 2020. Retrieved November 21, 2020.
Rutabingwa, Viviane; Kassaga Arinaitwe, James (September 1, 2015). "Taylor Swift Is Dreaming of a Very White Africa". NPR. Archived from the original on September 2, 2015. Retrieved September 2, 2015.
Duca, Lauren (August 30, 2015). "Taylor Swift's 'Wildest Dreams' Channels White Colonialism". HuffPost. Archived from the original on September 1, 2015. Retrieved September 2, 2015.
Mokoena, Tshepo (September 2, 2015). "Is Taylor Swift's Colonial Fantasy the Beginning of the End?". The Guardian. Archived from the original on December 29, 2023. Retrieved December 29, 2023.
Carotenuto, Matthew (September 23, 2015). "Taylor Swift's White Colonial Romance". JSTOR. Archived from the original on December 29, 2023. Retrieved December 29, 2023.
Keim & Somerville 2021, p. 149.
"Taylor Swift Video Director Defends 'Wildest Dreams' Following 'Whitewash' Claims". The Guardian. September 3, 2015. Archived from the original on January 14, 2021. Retrieved November 21, 2020.
Charlton, Lauretta (September 4, 2015). "Take a Breath – the 'Wildest Dreams' Video Isn't Racist – Now Exhale". Vulture. Archived from the original on August 5, 2017. Retrieved September 28, 2015.
Wood, Lucy (May 6, 2015). "Taylor Swift Has Kicked Off Her 1989 World Tour in Tokyo". Cosmopolitan. Archived from the original on June 14, 2018. Retrieved January 4, 2020.
Allen, Paige (July 25, 2015). "Review: Taylor Swift Delivers Another Stellar Show at Gillette". The Sun Chronicle. Archived from the original on May 31, 2022. Retrieved December 29, 2023.
Princiotti, Nora (March 16, 2023). "On the Eve of Eras, Ranking Taylor Swift's All-Time Best Live Performances". The Ringer. Archived from the original on March 31, 2023. Retrieved December 29, 2023.
"Taylor Swift Shares Stunning 'Wildest Dreams' Performance from Grammy Museum". Billboard. January 4, 2016. Archived from the original on November 29, 2020. Retrieved November 21, 2020.
Akers, Trenton; Arnold, Rikki-Lee (December 4, 2015). "Taylor Swift 1989 Tour: Exclusive Picture as Taylor Swift Arrives In Brisbane Ahead of Concert". The Courier-Mail. Archived from the original on December 29, 2023. Retrieved December 29, 2023.
Hall, David Brendan (October 23, 2016). "Taylor Swift Delivers a Knockout Performance at Formula 1 Concert in Austin". Billboard. Archived from the original on September 2, 2022. Retrieved December 29, 2023.
Atkinson, Katie (February 5, 2017). "Taylor Swift Performs 'Better Man' & 'I Don't Wanna Live Forever' for First Time at Stunning Pre-Super Bowl Set". Billboard. Archived from the original on May 25, 2022. Retrieved December 29, 2023.
Iasimone, Ashley (November 20, 2018). "All the Surprise Songs Taylor Swift Has Performed On Her Reputation Stadium Tour B-Stage (So Far)". Billboard. Archived from the original on October 8, 2019. Retrieved December 19, 2018.
Fisher, Luchina (July 16, 2018). "Taylor Swift Turns a Concert Malfunction Into a Memorable Moment". Good Morning America. Archived from the original on December 29, 2023. Retrieved December 29, 2023.
Yahr, Emily (March 18, 2023). "Taylor Swift's Eras Tour Opener: A Complete Recap of All 44 Songs". The Washington Post. Archived from the original on March 18, 2023. Retrieved March 18, 2023.
Jones, Nate (September 17, 2015). "Ryan Adams Is Finally Releasing His 1989 Covers Album; Listen to His 'Bad Blood'". Vulture. Archived from the original on December 29, 2023. Retrieved December 29, 2023.
Browne, David (September 21, 2015). "Ryan Adams on His Full-Album Cover of Taylor Swift's 1989". Rolling Stone. Archived from the original on September 25, 2023. Retrieved December 29, 2023.
Goodman, Jessica (August 13, 2015). "Ryan Adams Tackles Taylor Swift's 'Wildest Dreams' in Latest 1989 Cover". Entertainment Weekly. Archived from the original on December 29, 2023. Retrieved December 29, 2023.
Caffrey, Dam (September 28, 2015). "Ryan Adams – 1989". Consequence of Sound. Archived from the original on October 28, 2020. Retrieved September 28, 2015.
Burrows, Marc (October 30, 2015). "Album Review: Ryan Adams - 1989". Drowned in Sound. Archived from the original on December 29, 2023. Retrieved December 29, 2023.
Winograd, Jeremy (October 21, 2015). "Review: Ryan Adams, 1989". Slant Magazine. Archived from the original on May 6, 2019. Retrieved December 29, 2023.
Murphy, Sarah (September 22, 2015). "Ryan Adams 1989". Exclaim!. Archived from the original on November 3, 2023. Retrieved December 29, 2023.
Wood, Mikael (September 21, 2015). "Review: Ryan Adams Turns to Taylor Swift for Help On His Version of 1989". Los Angeles Times. Archived from the original on September 23, 2015. Retrieved September 21, 2015.
Sawdy, Evan (September 24, 2015). "Ryan Adams: 1989". PopMatters. Archived from the original on August 2, 2018. Retrieved September 24, 2015.
Cragg, Michael (September 22, 2015). "Ryan Adams's Take on Taylor Swift's 1989 – First Listen Track-By-Track Review". The Guardian. Archived from the original on January 28, 2023. Retrieved December 29, 2023.
"Ryan Adams Chart History (Hot Rock & Alternative Songs)". Billboard. Archived from the original on November 13, 2023. Retrieved December 29, 2023.
Kornharber, Spencer (September 21, 2015). "Ryan Adams's 1989 and the Vindication of Taylor Swift". The Atlantic. Archived from the original on November 8, 2020. Retrieved November 21, 2020.
Aroesti, Rachel (November 5, 2015). "Ryan Adams: 1989 Review – False Notes Abound on Taylor Swift Covers Album". The Guardian. Archived from the original on November 29, 2022. Retrieved December 29, 2023.
Caramanica, Jon (September 22, 2015). "Teaming Up, Together (Drake and Future) or Apart (Ryan Adams and Taylor Swift)". The New York Times. Archived from the original on September 24, 2015. Retrieved December 29, 2023.
"Taylor Swift – Wildest Dreams" (in German). Ö3 Austria Top 40. Retrieved November 18, 2015.
"Taylor Swift – Wildest Dreams" (in Dutch). Ultratop 50. Retrieved October 31, 2015.
"Taylor Swift – Wildest Dreams" (in French). Ultratip. Retrieved September 18, 2015.
"Taylor Swift Chart History (Canada AC)". Billboard. Retrieved December 13, 2023.
"Taylor Swift Chart History (Canada CHR/Top 40)". Billboard. Retrieved October 13, 2015.
"Taylor Swift Chart History (Canada Hot AC)". Billboard. Retrieved October 13, 2015.
"ČNS IFPI" (in Czech). Hitparáda – Radio Top 100 Oficiální. IFPI Czech Republic. Note: Select 44. týden 2015 in the date selector. Retrieved November 2, 2015.
"Taylor Swift: Wildest Dreams" (in Finnish). Musiikkituottajat. Retrieved June 25, 2019.
"Taylor Swift: Wildest Dreams" (in Finnish). Musiikkituottajat. Retrieved August 4, 2016.
"Taylor Swift – Wildest Dreams" (in French). Les classement single. Retrieved September 17, 2015.
"Taylor Swift Chart History (Greece Digital Song Sales)". Billboard. Archived from the original on December 6, 2019. Retrieved November 9, 2021.
"Archívum – Slágerlisták – MAHASZ" (in Hungarian). Single (track) Top 40 lista. Magyar Hanglemezkiadók Szövetsége. Retrieved September 10, 2015.
"Irish-charts.com – Discography Taylor Swift". Irish Singles Chart. Retrieved January 29, 2020.
"The official lebanese Top 20 – Taylor Swift". The Official Lebanese Top 20. Archived from the original on September 17, 2016. Retrieved September 1, 2016.
"Taylor Swift Chart History (Mexico Airplay)". Billboard. Archived from the original on August 19, 2019. Retrieved February 22, 2021.
"Top 20 Inglés Del 9 al 15 de Noviembre, 2015". Monitor Latinoaccessdate=2018-05-02. November 9, 2015.
"Tipparade-lijst van week 42, 2015" (in Dutch). Dutch Top 40. Archived from the original on March 19, 2023. Retrieved March 19, 2023.
"Listy bestsellerów, wyróżnienia :: Związek Producentów Audio-Video". Polish Airplay Top 100. Retrieved December 28, 2015.
"SloTop50 – Slovenian official singles chart". slotop50.si. Archived from the original on July 18, 2018. Retrieved July 18, 2018.
"Taylor Swift: Artist Chart History". Official Charts Company. Retrieved October 23, 2015.
"Taylor Swift Chart History (Hot 100)". Billboard. Retrieved December 13, 2023.
"Taylor Swift Chart History (Adult Contemporary)". Billboard. Retrieved December 13, 2023.
"Taylor Swift Chart History (Adult Pop Songs)". Billboard. Retrieved October 20, 2015.
"Taylor Swift Chart History (Dance Mix/Show Airplay)". Billboard. Retrieved November 3, 2015.
"Taylor Swift Chart History (Pop Songs)". Billboard. Retrieved October 27, 2015.
"Taylor Swift Chart History (Rhythmic)". Billboard. Retrieved November 3, 2015.
"ČNS IFPI" (in Czech). Hitparáda – Digital Top 100 Oficiální. IFPI Czech Republic. Note: Select 39. týden 2021 in the date selector. Retrieved October 4, 2021.
"Taylor Swift – Wildest Dreams" (in German). GfK Entertainment charts. Retrieved September 24, 2021.
"Taylor Swift Chart History (Global 200)". Billboard. Retrieved December 13, 2023.
"Taylor Swift – Wildest Dreams". AFP Top 100 Singles. Retrieved September 30, 2021.
"ČNS IFPI". IFPI ČR. Note: Select SK SINGLES DIGITAL TOP 100 and insert 202139 into search. Archived from the original on April 1, 2019. Retrieved October 4, 2021.
"Taylor Swift – Wildest Dreams". Singles Top 100. Retrieved November 6, 2023.
"Taylor Swift – Wildest Dreams". Swiss Singles Chart. Retrieved October 3, 2021.
"ARIA Charts – End of Year Charts – Top 100 Singles 2015". Australian Recording Industry Association. Archived from the original on January 24, 2016. Retrieved January 6, 2016.
"Canadian Hot 100 Year End 2015". Billboard. January 2, 2013. Archived from the original on December 11, 2015. Retrieved December 11, 2015.
"Hot 100: Year End 2015". Billboard. Archived from the original on December 21, 2015. Retrieved December 9, 2015.
"Adult Contemporary Songs – Year-End 2015". Billboard. January 2, 2013. Archived from the original on November 6, 2019. Retrieved October 2, 2019.
"Adult Pop Songs – Year-End 2015". Billboard. January 2, 2013. Archived from the original on May 18, 2019. Retrieved October 2, 2019.
"Pop Songs: Year-End 2015". Billboard. January 2, 2013. Archived from the original on May 18, 2019. Retrieved April 29, 2019.
"Canadian Hot 100 Year End 2016". Billboard. January 2, 2013. Archived from the original on December 21, 2016. Retrieved December 9, 2016.
"Hot 100 Songs – Year-End 2016". Billboard. Archived from the original on January 26, 2017. Retrieved December 8, 2016.
"Adult Contemporary Songs: Year-End 2016". Billboard. January 2, 2013. Archived from the original on December 11, 2016. Retrieved December 18, 2016.
"Adult Pop Songs: Year-End 2016". Billboard. January 2, 2013. Archived from the original on December 22, 2016. Retrieved December 18, 2016.
"New Zealand single certifications – Taylor Swift – Wildest Dreams". Radioscope. Retrieved December 19, 2024. Type Wildest Dreams in the "Search:" field.
"American single certifications – Taylor Swift – Wildest Dreams". Recording Industry Association of America. Retrieved March 13, 2020.
Melas, Chloe (November 16, 2020). "Taylor Swift Speaks Out About Sale of Her Masters". CNN. Archived from the original on November 18, 2020. Retrieved November 19, 2020.
"Taylor Swift Wants to Re-Record Her Old Hits". BBC News. August 22, 2019. Archived from the original on August 22, 2019. Retrieved August 22, 2019.
Finnis, Alex (November 17, 2020). "Taylor Swift Masters: The Controversy around Scooter Braun Selling the Rights to Her Old Music Explained". i. Archived from the original on February 12, 2021. Retrieved February 13, 2021.
Shah, Neil (April 9, 2021). "Taylor Swift Releases New Fearless Album, Reclaiming Her Back Catalog". The Wall Street Journal. Archived from the original on October 8, 2021. Retrieved September 25, 2022.
Espada, Mariah (July 6, 2023). "Taylor Swift Is Halfway Through Her Rerecording Project. It's Paid Off Big Time". Time. Archived from the original on October 27, 2023. Retrieved November 6, 2023.
Fernández, Alexia (March 12, 2021). "Spirit Untamed First Look! Hear Taylor Swift's Re-Recorded 'Wildest Dreams (Taylor's Version)' in Trailer". People. Archived from the original on March 12, 2021. Retrieved March 12, 2021.
Moore, Sam (April 1, 2021). "Listen to a New Preview of Taylor Swift's 'Wildest Dreams (Taylor's Version)'". NME. Archived from the original on August 11, 2023. Retrieved December 29, 2023.
Kenneally, Cerys (May 17, 2021). "Extended Clip of 'Wildest Dreams (Taylor's Version)' Features in New Spirit Untamed Trailer". The Line of Best Fit. Archived from the original on November 28, 2021. Retrieved May 17, 2021.
Speakman, Kimberlee (September 17, 2021). "Taylor Swift Drops New Version Of 'Wildest Dreams'—Why It Matters". Forbes. Archived from the original on August 10, 2023. Retrieved December 29, 2023.
Legatspi, Althea (September 17, 2021). "Taylor Swift Surprise-Releases 'Wildest Dreams (Taylor's Version)' for Avid TikTokers". Rolling Stone. Archived from the original on November 14, 2021. Retrieved December 29, 2023.
Vassell, Nicole (October 27, 2023). "Taylor Swift Fans Celebrate As Pop Star Releases 1989 (Taylor's Version)". The Independent. Archived from the original on October 30, 2023. Retrieved December 29, 2023.
Siroky, Mary (September 17, 2021). "Song of the Week: Taylor Swift Revives Our 'Wildest Dreams' with Surprise Re-Recording". Consequence. Archived from the original on May 6, 2022. Retrieved May 6, 2022.
Willman, Chris (September 17, 2021). "Taylor Swift Releases New Version of 'Wildest Dreams' From 1989, Skipping Ahead in Her Re-Recordings". Variety. Archived from the original on November 4, 2023. Retrieved December 29, 2023.
Aroesti, Rachel (October 27, 2023). "Taylor Swift: 1989 (Taylor's Version) Review – Subtle Bonus Tracks Add New Depths to a Classic". The Guardian. Retrieved December 29, 2023.
Murray, Robin (September 17, 2021). "Taylor Swift Just Shared 'Wildest Dreams (Taylor's Version)'". Clash. Archived from the original on December 3, 2021. Retrieved December 29, 2023.
Willman, Chris (September 17, 2021). "Taylor Swift's 'Wildest Dreams (Taylor's Version)' Quickly Beats the Original Song's Spotify Record for Single-Day Plays". Variety. Archived from the original on September 17, 2021. Retrieved September 17, 2021.
Zeliner, Xander (September 27, 2021). "Taylor Swift's 'Wildest Dreams (Taylor's Version)' Debuts in Hot 100's Top 40". Billboard. Archived from the original on November 14, 2021. Retrieved September 28, 2021.
Knopper, Steve (August 17, 2022). "How a Kid Flick Got Taylor Swift to Remake a Previously Off-Limits Song". Billboard. Archived from the original on August 18, 2022. Retrieved December 25, 2024.
White, Jack (September 24, 2021). "Ed Sheeran's 'Shivers' Holds Firm At Irish Number 1". Official Charts Company. Archived from the original on September 24, 2021. Retrieved September 24, 2021.
Copsey, Rob (September 24, 2021). "Ed Sheeran's 'Shivers' Scores Second Week at Official Singles Chart Summit". Official Charts Company. Archived from the original on September 24, 2021. Retrieved September 24, 2021.
Zellner, Xander (November 6, 2023). "Taylor Swift Charts All 21 Songs From 1989 (Taylor's Version) on the Hot 100". Billboard. Archived from the original on November 6, 2023. Retrieved December 29, 2023.
"Top 20 Most Streamed International + Domestic Songs Week 38 (17/09/2021-23/09/2021)". Recording Industry Association of Malaysia. October 2, 2021. Archived from the original on February 7, 2023. Retrieved October 2, 2021.
"RIAS Top Charts Week 38 (17–23 September 2021)". Recording Industry Association Singapore. Archived from the original on September 29, 2021. Retrieved September 29, 2021.
"Taylor Swift – Wildest Dreams (Taylor's Version)". ARIA Top 50 Singles. Retrieved November 6, 2023.
"Taylor Swift Chart History (Philippines Songs)". Billboard. Archived from the original on November 8, 2023. Retrieved November 7, 2023.
"Archívum – Slágerlisták – MAHASZ" (in Hungarian). Single (track) Top 40 lista. Magyar Hanglemezkiadók Szövetsége. Retrieved September 30, 2021.
"Taylor Swift – Wildest Dreams (Taylor's Version)". Top 40 Singles. Retrieved September 24, 2021.
"New Zealand single certifications – Taylor Swift – Wildest Dreams (Taylor's Version)". Recorded Music NZ. Retrieved November 20, 2024.
"Digital Singles Chart (International)". IFPI Greece. Archived from the original on November 13, 2023. Retrieved November 8, 2023.
"British single certifications – Taylor Swift – Wildest Dreams (Taylor's Version)". British Phonographic Industry. Retrieved January 8, 2024.
"Taylor Swift Chart History (Euro Digital Song Sales)". Billboard. Retrieved November 18, 2021.
"Taylor Swift – Wildest Dreams (Taylor's Version)" (in French). Les classement single. Retrieved November 16, 2023.
"Official Irish Singles Chart Top 50". Official Charts Company. Retrieved September 24, 2021.
"2021 39-os savaitės klausomiausi (Top 100)" (in Lithuanian). AGATA. October 1, 2021. Archived from the original on October 1, 2021. Retrieved October 1, 2021.
"Dutch Single Top 100". Hung Medien. Archived from the original on November 8, 2023. Retrieved November 8, 2023.
"Local & International Streaming Chart Top 100: Week 38". Recording Industry of South Africa. Archived from the original on September 30, 2021. Retrieved October 1, 2021.
"Taylor Swift – Wildest Dreams (Taylor's Version)". Singles Top 100. Retrieved November 6, 2023.
"Official Singles Chart Top 100". Official Charts Company. Retrieved September 24, 2021.
"Taylor Swift Chart History (Billboard Vietnam Hot 100)". Billboard. Archived from the original on June 6, 2022. Retrieved November 11, 2023.
"ARIA Charts – Accreditations – 2024 Singles" (PDF). Australian Recording Industry Association. Retrieved February 14, 2024.

    "OLiS - oficjalna lista wyróżnień" (in Polish). Polish Society of the Phonographic Industry. Retrieved September 11, 2024. Click "TYTUŁ" and enter Wildest Dreams (Taylor's Version) in the search box.

Sources

    Dyer, Elizabeth B. (December 19, 2016). "Whitewashed African Film Sets: Taylor Swift's 'Wildest Dreams' and King Solomon's Mines". African Studies Review. 59 (3): 301–310. doi:10.1017/asr.2016.93. S2CID 229168394.
    Keim, Curtis; Somerville, Carolyn (2021). "Safari: Beyond Our Wildest Dreams". Mistaking Africa: Misconceptions and Inventions. Taylor & Francis. doi:10.4324/9781003172024-12. ISBN 978-1-000-51001-0.
    McNutt, Myles (2020). "From 'Mine' to 'Ours': Gendered Hierarchies of Authorship and the Limits of Taylor Swift's Paratextual Feminism". Communication, Culture and Critique. 13 (1): 72–91. doi:10.1093/ccc/tcz042.
    Perone, James E. (2017). "1989 and Beyond". The Words and Music of Taylor Swift. The Praeger Singer-Songwriter Collection. ABC-Clio. pp. 55–68. ISBN 978-1-44-085294-7.

    vte

Taylor Swift songs

    Singles discography

Taylor Swift	

    "Tim McGraw" "Picture to Burn" "Teardrops on My Guitar" "A Place in This World" "Should've Said No" "Our Song"

Fearless	

    "Fearless" "Fifteen" "Love Story" "Hey Stephen" "White Horse" "You Belong with Me" "Breathe" "You're Not Sorry" "The Way I Loved You" "Forever & Always" "The Best Day" "Change"

Taylor's Version	

    "You All Over Me" "Mr. Perfectly Fine" "That's When"

Speak Now	

    "Mine" "Sparks Fly" "Back to December" "Speak Now" "Dear John" "Mean" "The Story of Us" "Never Grow Up" "Enchanted" "Better than Revenge" "Innocent" "Haunted" "Long Live" "Ours"

Taylor's Version	

    "Electric Touch" "When Emma Falls in Love" "I Can See You" "Castles Crumbling"

Red	

    "State of Grace" "Red" "Treacherous" "I Knew You Were Trouble" "All Too Well" "22" "We Are Never Ever Getting Back Together" "The Last Time" "Holy Ground" "Everything Has Changed" "Begin Again"

Taylor's Version	

    "Nothing New" "Message in a Bottle" "I Bet You Think About Me" "Forever Winter"

1989	

    "Welcome to New York" "Blank Space" "Style" "Out of the Woods" "All You Had to Do Was Stay" "Shake It Off" "I Wish You Would" "Bad Blood" "Wildest Dreams" "How You Get the Girl" "This Love" "I Know Places" "Clean" "You Are in Love" "New Romantics"

Taylor's Version	

    "'Slut!'" "Say Don't Go" "Now That We Don't Talk" "Suburban Legends" "Is It Over Now?"

Reputation	

    "...Ready for It?" "End Game" "I Did Something Bad" "Don't Blame Me" "Delicate" "Look What You Made Me Do" "So It Goes..." "Gorgeous" "Getaway Car" "Dress" "Call It What You Want" "New Year's Day"

Lover	

    "I Forgot That You Existed" "Cruel Summer" "Lover" "The Man" "The Archer" "Miss Americana & the Heartbreak Prince" "Paper Rings" "Cornelia Street" "Death by a Thousand Cuts" "London Boy" "Soon You'll Get Better" "False God" "You Need to Calm Down" "Me!"

Folklore	

    "The 1" "Cardigan" "The Last Great American Dynasty" "Exile" "My Tears Ricochet" "Mirrorball" "Seven" "August" "This Is Me Trying" "Illicit Affairs" "Invisible String" "Mad Woman" "Epiphany" "Betty" "Peace" "Hoax" "The Lakes"

Evermore	

    "Willow" "Champagne Problems" "'Tis the Damn Season" "Tolerate It" "No Body, No Crime" "Happiness" "Dorothea" "Coney Island" "Long Story Short" "Marjorie"

Midnights	

    "Lavender Haze" "Maroon" "Anti-Hero" "Snow on the Beach" "You're on Your Own, Kid" "Midnight Rain" "Question...?" "Vigilante Shit" "Bejeweled" "Labyrinth" "Karma" "Sweet Nothing" "Mastermind" "Hits Different" "Bigger Than the Whole Sky" "Would've, Could've, Should've" "You're Losing Me"

The Tortured Poets
Department	

    "Fortnight" "The Tortured Poets Department "My Boy Only Breaks His Favorite Toys" "Down Bad" "So Long, London" "But Daddy I Love Him" "Fresh Out the Slammer" "Florida!!!" "Guilty as Sin?" "Who's Afraid of Little Old Me?" "Loml" "I Can Do It with a Broken Heart" "The Smallest Man Who Ever Lived" "The Alchemy" "Clara Bow" "The Black Dog" "So High School" "Thank You Aimee"

Soundtrack songs	

    "Crazier" "Today Was a Fairytale" "Safe & Sound" "Eyes Open" "Sweeter than Fiction" "I Don't Wanna Live Forever" "Beautiful Ghosts" "Only the Young" "Carolina"

Featured songs	

    "Two Is Better Than One" "Half of My Heart" "Both of Us" "Babe" "Gasoline" "Renegade" "The Joker and the Queen" "The Alcott" "Us"

Other songs	

    "Best Days of Your Life" "Ronan" "Highway Don't Care" "Better Man" "Christmas Tree Farm" "All of the Girls You Loved Before"

    Category

Authority control databases Edit this at Wikidata	

    MusicBrainz workMusicBrainz release group
        2

Categories:

    2010s ballads2014 songs2015 singlesTaylor Swift songsSong recordings produced by Max MartinSong recordings produced by Shellback (record producer)Song recordings produced by Taylor SwiftSong recordings produced by Chris RoweSongs written by Taylor SwiftSongs written by Max MartinSongs written by Shellback (record producer)Music videos directed by Joseph KahnDream pop songsAmerican synth-pop songsElectropop balladsSynth-pop balladsRyan Adams songsBig Machine Records singlesRepublic Records singlesMusic video controversiesRace-related controversies in musicSongs about dreams

    This page was last edited on 25 December 2024, at 01:47 (UTC).
    Text is available under the Creative Commons Attribution-ShareAlike 4.0 License; additional terms may apply. By using this site, you agree to the Terms of Use and Privacy Policy. Wikipedia® is a registered trademark of the Wikimedia Foundation, Inc., a non-profit organization.

    Privacy policy
    About Wikipedia
    Disclaimers
    Contact Wikipedia
    Code of Conduct
    Developers
    Statistics
    Cookie statement
    Mobile view

    Wikimedia Foundation
    Powered by MediaWiki


"""

top_n = 5000  # Number of top recommendations to keep
final_top_songs = []

# Process the data in batches and update recommendations incrementally
print("Recommending songs incrementally...")
start = time.time()
for index, batch_data in enumerate(load_data_in_batches(db_path)):
    processed_batch = preprocess_data(batch_data)
    vectorized_batch = vectorize_data(processed_batch, vectorizer)
    
    recommended_songs = recommend_songs(preprocess_article(my_article), vectorized_batch, vectorizer, top_n=top_n)
    
    # Maintain the top_n songs by combining and sorting
    final_top_songs.extend(recommended_songs)
    final_top_songs = sorted(final_top_songs, key=lambda x: x[1], reverse=True)[:top_n]
    
    print(f"{time.time() - start} seconds. Batch #{index} out of {sum(1 for _ in load_data_in_batches(db_path))}")
    print()

print("Final recommended songs:")
for index, (song, score) in enumerate(final_top_songs):
    print(f"{index}. Song: {song.ljust(100)} Similarity: {score}")
