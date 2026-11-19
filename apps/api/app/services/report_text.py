"""Words for the family report, in English and Hindi.

Exam and degree names (NEET-UG, MBBS, CUET-UG...) stay in English in both
versions, because that is how they appear on official forms and websites.
"""

from typing import Literal

Lang = Literal["en", "hi"]

TEXT: dict[Lang, dict[str, str]] = {
    "en": {
        "title": "Career report for {name}",
        "intro": "A summary of what {name} enjoys, the careers that fit best, and the steps to get there. "
        "It's meant to start a family conversation, not to make the decision.",
        "interests": "What {name} enjoys",
        "interests_none": "{name} hasn't taken the interest questionnaire yet.",
        "careers": "Careers that fit best",
        "careers_none": "Careers appear here once {name} finishes the interest questionnaire.",
        "why": "Why it fits",
        "keep_in_mind": "Keep in mind",
        "route": "A route from where {name} is now",
        "not_open": "This route isn't open from {name}'s current stream.",
        "years": "About {years} years of study and training",
        "check_with": "Check the latest rules with",
        "tryouts": "Tried in practice",
        "tryouts_none": "{name} hasn't tried any job tryouts yet. They are short tasks from real jobs, on Pathify.",
        "tried_this": "Tried “{title}”",
        "questions": "Questions to talk about together",
        "about_heading": "About this report",
        "about": "This report comes from {name}'s answers to an interest questionnaire based on Holland's six "
        "interest types (RIASEC), and from short job tryouts on Pathify. Interests change as young people try "
        "new things, and exam rules change every year, so please confirm details with the official bodies listed. "
        "No college, coaching centre or company paid to appear here.",
        "names_note": "",
        "generated": "Prepared on {date}",
        "confidence_high": "These results are fairly settled.",
        "confidence_medium": "These results are a good starting point; a few careers are close together.",
        "confidence_low": "These results could still change; several interests are close together.",
        "enjoyment": "Enjoyed it: {n} of 5",
        "see_yes": "Could see themselves doing this",
        "see_maybe": "Maybe could see themselves doing this",
        "see_no": "Doesn't see themselves doing this",
        "fit": "{label}",
    },
    "hi": {
        "title": "{name} की करियर रिपोर्ट",
        "intro": "{name} को क्या पसंद है, कौन-से करियर सबसे ज़्यादा मेल खाते हैं, और वहाँ तक पहुँचने के कदम। "
        "यह रिपोर्ट परिवार में बातचीत शुरू करने के लिए है, फ़ैसला करने के लिए नहीं।",
        "interests": "{name} को क्या पसंद है",
        "interests_none": "{name} ने अभी रुचि वाली प्रश्नावली नहीं भरी है।",
        "careers": "सबसे ज़्यादा मेल खाने वाले करियर",
        "careers_none": "{name} के प्रश्नावली पूरी करने के बाद करियर यहाँ दिखेंगे।",
        "why": "क्यों मेल खाता है",
        "keep_in_mind": "ध्यान रखें",
        "route": "{name} अभी जहाँ हैं, वहाँ से एक रास्ता",
        "not_open": "{name} की मौजूदा स्ट्रीम से यह रास्ता खुला नहीं है।",
        "years": "लगभग {years} साल की पढ़ाई और ट्रेनिंग",
        "check_with": "ताज़ा नियम यहाँ देखें",
        "tryouts": "करके देखा",
        "tryouts_none": "{name} ने अभी कोई जॉब ट्रायआउट नहीं किया है। ये Pathify पर असली कामों के छोटे अभ्यास हैं।",
        "tried_this": "“{title}” करके देखा",
        "questions": "मिलकर बात करने के लिए सवाल",
        "about_heading": "इस रिपोर्ट के बारे में",
        "about": "यह रिपोर्ट {name} के रुचि-प्रश्नावली के जवाबों (हॉलैंड के छह रुचि प्रकार, RIASEC) और Pathify पर किए "
        "गए छोटे जॉब ट्रायआउट्स पर आधारित है। नई चीज़ें आज़माने के साथ रुचियाँ बदलती हैं, और परीक्षाओं के नियम हर साल "
        "बदलते हैं, इसलिए दी गई आधिकारिक संस्थाओं से जानकारी ज़रूर जाँच लें। किसी कॉलेज, कोचिंग या कंपनी ने यहाँ "
        "दिखने के लिए पैसे नहीं दिए हैं।",
        "names_note": "परीक्षा और डिग्री के नाम अंग्रेज़ी में हैं, जैसे वे सरकारी फ़ॉर्म और वेबसाइट पर लिखे होते हैं।",
        "generated": "{date} को तैयार की गई",
        "confidence_high": "ये नतीजे काफ़ी पक्के हैं।",
        "confidence_medium": "ये नतीजे शुरुआत के लिए अच्छे हैं; कुछ करियर एक-दूसरे के काफ़ी क़रीब हैं।",
        "confidence_low": "ये नतीजे अभी बदल सकते हैं; कई रुचियाँ एक-दूसरे के क़रीब हैं।",
        "enjoyment": "कितना पसंद आया: 5 में से {n}",
        "see_yes": "ख़ुद को यह काम करते हुए देख सकते हैं",
        "see_maybe": "शायद ख़ुद को यह काम करते हुए देख सकते हैं",
        "see_no": "ख़ुद को यह काम करते हुए नहीं देखते",
        "fit": "{label}",
    },
}

FIT_LABELS: dict[Lang, dict[str, str]] = {
    "en": {"Strong fit": "Strong fit", "Good fit": "Good fit", "Possible fit": "Possible fit", "Weak fit": "Weak fit"},
    "hi": {"Strong fit": "बहुत अच्छा मेल", "Good fit": "अच्छा मेल", "Possible fit": "संभव मेल", "Weak fit": "कम मेल"},
}

DIMENSIONS: dict[Lang, dict[str, tuple[str, str]]] = {
    "en": {
        "R": ("Hands-on", "building, fixing and working with tools, machines, plants or animals"),
        "I": ("Curious", "asking why, solving problems, and working with ideas, data and science"),
        "A": ("Creative", "making things that express ideas: design, writing, music, film, performance"),
        "S": ("Helping", "teaching, caring for and working closely with people"),
        "E": ("Leading", "leading, persuading and taking charge of plans and teams"),
        "C": ("Organising", "keeping records, numbers, schedules and rules accurate and in order"),
    },
    "hi": {
        "R": ("हाथ से काम", "औज़ारों, मशीनों, पौधों या जानवरों के साथ बनाना और ठीक करना"),
        "I": ("जिज्ञासु", "सवाल पूछना, समस्याएँ सुलझाना, और विचारों, डेटा व विज्ञान के साथ काम करना"),
        "A": ("रचनात्मक", "डिज़ाइन, लेखन, संगीत, फ़िल्म या प्रदर्शन के ज़रिए कुछ नया बनाना"),
        "S": ("मदद करना", "सिखाना, देखभाल करना और लोगों के साथ मिलकर काम करना"),
        "E": ("नेतृत्व", "नेतृत्व करना, दूसरों को मनाना और योजनाओं व टीमों की ज़िम्मेदारी लेना"),
        "C": ("व्यवस्थित", "रिकॉर्ड, संख्याएँ, समय-सारणी और नियम सही और व्यवस्थित रखना"),
    },
}

ACTIVITY: dict[Lang, dict[str, str]] = {
    "en": {
        "R": "hands-on work with tools, machines or nature",
        "I": "investigating and solving problems",
        "A": "creative work",
        "S": "helping and teaching people",
        "E": "leading and persuading",
        "C": "organising details and records",
    },
    "hi": {
        "R": "हाथ से काम (औज़ार, मशीन या प्रकृति के साथ)",
        "I": "खोज करना और समस्याएँ सुलझाना",
        "A": "रचनात्मक काम",
        "S": "लोगों की मदद करना और सिखाना",
        "E": "नेतृत्व करना और दूसरों को मनाना",
        "C": "बारीकियाँ और रिकॉर्ड व्यवस्थित रखना",
    },
}

REASONS: dict[Lang, dict[str, str]] = {
    "en": {
        "shared_strength": "{name} enjoys {activity}, and this career is full of it.",
        "shared_dislike": "It doesn't need much {activity}, which {name} is less keen on.",
        "career_needs_more": "It involves a lot of {activity}, which {name} rated low.",
        "career_uses_less": "{name} enjoys {activity}, but this career uses little of it.",
    },
    "hi": {
        "shared_strength": "{name} को {activity} पसंद है, और इस करियर में यह बहुत होता है।",
        # No postposition straight after {activity}: the phrases end in infinitives (करना), which would need
        # their oblique form (करने) before की.
        "shared_dislike": "इसमें ज़्यादा {activity} शामिल नहीं होता, जो {name} को कम पसंद है।",
        "career_needs_more": "इसमें {activity} बहुत होता है, जिसे {name} ने कम पसंद किया।",
        "career_uses_less": "{name} को {activity} पसंद है, लेकिन इस करियर में यह कम होता है।",
    },
}

STREAMS: dict[Lang, dict[str, str]] = {
    "en": {
        "science_pcm": "Science (PCM)",
        "science_pcb": "Science (PCB)",
        "science_pcmb": "Science (PCMB)",
        "commerce": "Commerce",
        "humanities": "Humanities",
        "vocational": "Vocational",
        "undecided": "undecided stream",
    },
    "hi": {
        "science_pcm": "विज्ञान (PCM)",
        "science_pcb": "विज्ञान (PCB)",
        "science_pcmb": "विज्ञान (PCMB)",
        "commerce": "कॉमर्स",
        "humanities": "कला / मानविकी",
        "vocational": "व्यावसायिक",
        "undecided": "स्ट्रीम अभी तय नहीं",
    },
}

STAGE: dict[Lang, dict[str, str]] = {
    "en": {"school": "Class {grade}", "college": "Studying {degree}", "graduate": "Graduate, {degree}"},
    "hi": {"school": "कक्षा {grade}", "college": "{degree} की पढ़ाई", "graduate": "स्नातक, {degree}"},
}

QUESTIONS: dict[Lang, dict[str, list[str]]] = {
    "en": {
        "school_9_10": [
            "Which of these careers made you curious, and why?",
            "Which subjects do you enjoy most at school, and which feel like a struggle?",
            "Which stream keeps most of these routes open?",
            "Who do we know who does one of these jobs, and could we talk to them?",
        ],
        "school_11_12": [
            "Which of these careers made you curious, and why?",
            "Which entrance exams would we need to plan for, and by when?",
            "What is our plan B if the first-choice exam doesn't go well?",
            "How many years of study does each route take, and does that work for our family?",
        ],
        "college": [
            "Which of these careers made you curious, and why?",
            "What skills or projects would make a first application stronger?",
            "Is more study needed, or is it better to start working first?",
            "Who do we know in this field who could share what the work is really like?",
        ],
    },
    "hi": {
        "school_9_10": [
            "इनमें से किस करियर के बारे में जानने की उत्सुकता हुई, और क्यों?",
            "स्कूल में कौन-से विषय सबसे अच्छे लगते हैं, और कौन-से मुश्किल लगते हैं?",
            "कौन-सी स्ट्रीम इनमें से ज़्यादातर रास्ते खुले रखती है?",
            "हम किसे जानते हैं जो इनमें से कोई काम करता है, क्या हम उनसे बात कर सकते हैं?",
        ],
        "school_11_12": [
            "इनमें से किस करियर के बारे में जानने की उत्सुकता हुई, और क्यों?",
            "किन प्रवेश परीक्षाओं की तैयारी करनी होगी, और कब तक?",
            "अगर पहली पसंद वाली परीक्षा अच्छी नहीं गई, तो हमारा प्लान B क्या है?",
            "हर रास्ते में कितने साल की पढ़ाई है, और क्या यह हमारे परिवार के लिए ठीक है?",
        ],
        "college": [
            "इनमें से किस करियर के बारे में जानने की उत्सुकता हुई, और क्यों?",
            "कौन-से कौशल या प्रोजेक्ट पहली नौकरी की अर्ज़ी को मज़बूत बनाएँगे?",
            "क्या आगे पढ़ाई ज़रूरी है, या पहले काम शुरू करना बेहतर है?",
            "इस क्षेत्र में हम किसे जानते हैं जो बता सके कि यह काम असल में कैसा है?",
        ],
    },
}
