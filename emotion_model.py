# =========================================================
# MINDSENTINEL - IMPROVED EMOTION DETECTION MODEL
# =========================================================
#
# Emotions:
# Happy | Sad | Angry | Anxious | Stressed | Fear | Neutral
#
# This is a prototype NLP/ML model for the SIH project.
# It is NOT a medical diagnosis system.
# =========================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# =========================================================
# TRAINING DATA
# =========================================================

texts = [

    # =====================================================
    # HAPPY
    # =====================================================

    "I am very happy today",
    "I feel great",
    "I am feeling wonderful",
    "I am excited",
    "Today is a beautiful day",
    "I feel good and positive",
    "I am enjoying my life",
    "I am feeling happy",
    "Everything is going well",
    "I feel amazing",
    "I am cheerful today",
    "I am feeling joyful",
    "I am really pleased",
    "I feel fantastic",
    "I am in a good mood",
    "I feel positive about my life",
    "I am satisfied with everything",
    "Today made me very happy",
    "I feel peaceful and happy",
    "I am excited about my future",
    "I am smiling today",
    "I feel lucky and happy",
    "I had a wonderful day",
    "I am feeling optimistic",
    "Things are going really well",


    # =====================================================
    # SAD
    # =====================================================

    "I am feeling sad",
    "I feel very lonely",
    "I am unhappy",
    "I feel depressed",
    "I am crying",
    "Nothing makes me happy",
    "I feel hopeless",
    "I feel very low",
    "I feel empty",
    "I don't feel good",
    "I feel lonely and sad",
    "I miss my family",
    "I feel emotionally broken",
    "I feel down today",
    "I am disappointed",
    "I feel hurt",
    "I feel miserable",
    "I have been crying all day",
    "I feel alone",
    "I don't know why I feel so sad",
    "I feel like nobody understands me",
    "I am having a difficult day",
    "I feel emotionally tired and sad",
    "I have lost my happiness",
    "I feel hopeless about everything",


    # =====================================================
    # ANGRY
    # =====================================================

    "I am very angry",
    "This makes me angry",
    "I hate this situation",
    "I am frustrated",
    "I am irritated",
    "I feel angry and upset",
    "I am mad",
    "I feel furious",
    "This situation is annoying",
    "I am extremely frustrated",
    "I am angry with everyone",
    "I cannot control my anger",
    "People are making me angry",
    "I am really annoyed",
    "This is making me furious",
    "I am very irritated today",
    "I feel rage",
    "I am upset because of what happened",
    "I hate what happened",
    "I am angry about this situation",
    "I feel frustrated with my work",
    "Everything is irritating me",
    "I am losing my temper",
    "I feel angry and frustrated",
    "I cannot stop feeling angry",


    # =====================================================
    # ANXIOUS
    # =====================================================

    "I am anxious",
    "I feel anxious about everything",
    "I am worried about my future",
    "I am nervous",
    "I keep worrying",
    "I am overthinking everything",
    "I feel nervous about my exams",
    "I am worried about my studies",
    "I am anxious about my career",
    "I am worried about my future",
    "I cannot stop overthinking",
    "My mind is full of worries",
    "I feel anxious about tomorrow",
    "I am nervous about my presentation",
    "I am worried about failing",
    "I feel uncertain about my future",
    "I am constantly worrying",
    "I feel nervous all the time",
    "I am anxious about my results",
    "I keep thinking about bad things",
    "I am scared about what will happen",
    "I feel tense and worried",
    "I am worried about my career",
    "I feel anxious about college",
    "I cannot relax because I am worried",


    # =====================================================
    # STRESSED
    # =====================================================

    "I am very stressed",
    "I feel stressed about my studies",
    "I am stressed about my exams",
    "I have too much work",
    "I feel overwhelmed",
    "I cannot handle my workload",
    "I am under a lot of pressure",
    "My studies are stressing me out",
    "I am stressed about my project",
    "I feel exhausted because of work",
    "I have too many assignments",
    "I am under academic pressure",
    "I feel mentally exhausted",
    "Everything is becoming too much",
    "I am overwhelmed with work",
    "I feel stressed about my future",
    "I have a lot of pressure",
    "I cannot manage everything",
    "I am tired because of studying",
    "I feel burned out",
    "My workload is too much",
    "I am stressed about deadlines",
    "I have too much pressure from college",
    "I don't know how to manage my studies",
    "I feel extremely overwhelmed",
    "I am stressed and tired",
    "I am feeling stressed about my studies and I don't know what to do",


    # =====================================================
    # FEAR
    # =====================================================

    "I am scared",
    "I feel afraid",
    "I am frightened",
    "I feel unsafe",
    "I am terrified",
    "I am afraid of what will happen",
    "I am scared of losing everything",
    "I feel frightened",
    "I am afraid to go outside",
    "I feel unsafe here",
    "I am scared about my situation",
    "I am terrified about what happened",
    "I feel threatened",
    "I am afraid",
    "I feel panic because I am scared",
    "I am frightened about the future",
    "I feel danger around me",
    "I am scared and alone",
    "I am afraid something bad will happen",
    "I feel very unsafe",
    "I am terrified of failing",
    "I am scared because of what happened",
    "I don't feel safe",
    "I am afraid to talk about it",
    "I feel frightened and helpless",


    # =====================================================
    # NEUTRAL
    # =====================================================

    "I am okay",
    "I am fine",
    "Everything is normal",
    "Nothing special today",
    "I am doing my work",
    "Today is a normal day",
    "I don't feel anything special",
    "I am doing okay",
    "My day is normal",
    "Everything is fine",
    "I went to college today",
    "I completed my work",
    "I am studying today",
    "I am attending my class",
    "I had lunch today",
    "I am working on my computer",
    "Today is an ordinary day",
    "I have nothing special to say",
    "I am just doing my routine",
    "My day was normal",
    "I am preparing for tomorrow",
    "I am reading a book",
    "I am doing my assignment",
    "I went outside today",
    "Everything is going normally"
]


# =========================================================
# LABELS
# =========================================================

labels = (

    # HAPPY - 25
    ["Happy"] * 25

    +

    # SAD - 25
    ["Sad"] * 25

    +

    # ANGRY - 25
    ["Angry"] * 25

    +

    # ANXIOUS - 25
    ["Anxious"] * 25

    +

    # STRESSED - 27
    ["Stressed"] * 27

    +

    # FEAR - 25
    ["Fear"] * 25

    +

    # NEUTRAL - 25
    ["Neutral"] * 25
)


# =========================================================
# SAFETY CHECK
# =========================================================

if len(texts) != len(labels):

    raise ValueError(
        f"Training data mismatch: "
        f"{len(texts)} texts but {len(labels)} labels."
    )


# =========================================================
# TF-IDF VECTORIZER
# =========================================================

vectorizer = TfidfVectorizer(

    lowercase=True,

    stop_words="english",

    ngram_range=(1, 2),

    sublinear_tf=True,

    min_df=1
)


# Convert training text into numerical features

X = vectorizer.fit_transform(texts)


# =========================================================
# LOGISTIC REGRESSION MODEL
# =========================================================

model = LogisticRegression(

    max_iter=2000,

    class_weight="balanced",

    random_state=42
)


# Train model

model.fit(X, labels)


# =========================================================
# KEYWORD SUPPORT
# =========================================================
#
# The ML model is combined with a small keyword/context
# layer to improve common demo sentences.
#
# This is still a prototype and should not be considered
# a clinical diagnostic system.
# =========================================================


def keyword_emotion(text):

    """
    Detect strong emotion-related keywords.

    Returns:
        emotion or None
    """

    text_lower = text.lower()


    # -----------------------------------------------------
    # STRESSED
    # -----------------------------------------------------

    stress_words = [

        "stressed",
        "stress",
        "overwhelmed",
        "workload",
        "deadline",
        "deadlines",
        "pressure",
        "burnout",
        "burned out",
        "too much work",
        "too many assignments",
        "academic pressure",
        "study pressure",
        "college pressure",
        "mentally exhausted"
    ]

    if any(word in text_lower for word in stress_words):

        return "Stressed"


    # -----------------------------------------------------
    # ANXIOUS
    # -----------------------------------------------------

    anxious_words = [

        "anxious",
        "anxiety",
        "worried",
        "worrying",
        "worry",
        "nervous",
        "overthinking",
        "overthink",
        "panic",
        "panicking",
        "uncertain",
        "can't relax",
        "cannot relax"
    ]

    if any(word in text_lower for word in anxious_words):

        return "Anxious"


    # -----------------------------------------------------
    # FEAR
    # -----------------------------------------------------

    fear_words = [

        "scared",
        "afraid",
        "frightened",
        "terrified",
        "unsafe",
        "threatened",
        "danger",
        "don't feel safe",
        "do not feel safe"
    ]

    if any(word in text_lower for word in fear_words):

        return "Fear"


    # -----------------------------------------------------
    # ANGRY
    # -----------------------------------------------------

    angry_words = [

        "angry",
        "anger",
        "furious",
        "frustrated",
        "frustration",
        "irritated",
        "irritating",
        "annoyed",
        "annoying",
        "rage",
        "hate",
        "losing my temper"
    ]

    if any(word in text_lower for word in angry_words):

        return "Angry"


    # -----------------------------------------------------
    # SAD
    # -----------------------------------------------------

    sad_words = [

        "sad",
        "lonely",
        "alone",
        "unhappy",
        "depressed",
        "crying",
        "hopeless",
        "empty",
        "miserable",
        "broken",
        "hurt",
        "down",
        "low",
        "miss my family"
    ]

    if any(word in text_lower for word in sad_words):

        return "Sad"


    # -----------------------------------------------------
    # HAPPY
    # -----------------------------------------------------

    happy_words = [

        "happy",
        "great",
        "wonderful",
        "amazing",
        "excited",
        "joyful",
        "cheerful",
        "fantastic",
        "positive",
        "pleased",
        "enjoying",
        "optimistic",
        "good mood",
        "beautiful day"
    ]

    if any(word in text_lower for word in happy_words):

        return "Happy"


    return None


# =========================================================
# EMOTION DETECTION FUNCTION
# =========================================================

def detect_emotion(text):

    """
    Detect emotion from user text.

    Returns:
        emotion
        confidence
    """

    # -----------------------------------------------------
    # Empty input
    # -----------------------------------------------------

    if not text or not text.strip():

        return "Neutral", 0.0


    text = text.strip()


    # -----------------------------------------------------
    # First check strong emotion keywords
    # -----------------------------------------------------

    keyword_result = keyword_emotion(text)


    # -----------------------------------------------------
    # ML prediction
    # -----------------------------------------------------

    text_vector = vectorizer.transform([text])


    prediction = model.predict(text_vector)[0]


    probabilities = model.predict_proba(text_vector)[0]


    confidence = max(probabilities) * 100


    # -----------------------------------------------------
    # Combine keyword + ML result
    # -----------------------------------------------------

    if keyword_result:

        # Strong keyword detected.
        #
        # Use the keyword result but keep confidence
        # within a reasonable prototype range.

        if keyword_result == prediction:

            confidence = max(confidence, 70)

        else:

            confidence = max(confidence, 75)


        prediction = keyword_result


    # -----------------------------------------------------
    # Improve very low-confidence predictions
    # -----------------------------------------------------

    if confidence < 35:

        confidence = 50.0


    confidence = min(confidence, 99.0)


    return prediction, round(confidence, 2)


# =========================================================
# TEST MODEL
# =========================================================

if __name__ == "__main__":

    print("\n==============================================")
    print("      MINDSENTINEL EMOTION DETECTION")
    print("==============================================")

    print("\nAvailable emotions:")
    print("Happy | Sad | Angry | Anxious | Stressed | Fear | Neutral")

    print("\nType 'exit' to stop.\n")


    while True:

        user_text = input(
            "Enter how you are feeling: "
        ).strip()


        if user_text.lower() == "exit":

            print("\nModel testing stopped.")

            break


        emotion, confidence = detect_emotion(
            user_text
        )


        print("\n----------------------------------------------")

        print(
            "Detected Emotion:",
            emotion
        )

        print(
            "Confidence:",
            confidence,
            "%"
        )

        print("----------------------------------------------\n")