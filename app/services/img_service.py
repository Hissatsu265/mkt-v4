import random
prompts_image_ex = [
    "A high-quality, realistic promotional poster that beautifully highlights the product in the input image. The composition looks professional and visually appealing, with lighting, shadows, and background elements that harmonize naturally with the product. The design should emphasize the product as the main focus, with elegant details, reflections, or subtle decorations that enhance its appearance. The style is clean, modern, and suitable for advertising use. No people in the scene.",
    "A realistic and elegant product showcase poster that focuses entirely on the item from the input image. The background is minimalistic and refined, with balanced lighting, soft shadows, and subtle reflections that make the product stand out beautifully. The style is clean, premium, and suitable for high-end advertising. No people or distracting elements.",
    "A cinematic, high-quality promotional image featuring the product as the central subject. The lighting and shadows are carefully designed to create depth and highlight the product’s texture and form. The background complements the color tone of the product, adding a touch of artistic elegance. No humans in the frame.",
    "A modern, realistic advertising poster displaying the product attractively against a stylish background. The layout is clean and professional, with soft gradients or geometric accents that enhance the product’s presence. Lighting and reflections are natural and balanced. No people included.",
    "A realistic and elegant flat-lay promotional poster showing the product from a top-down perspective. The composition is clean and balanced, with natural lighting, soft shadows, and subtle reflections that highlight the product’s design and details. The background is minimalistic, ensuring the product remains the main focus. No people in the scene.",
    "A realistic promotional photo of the product placed outdoors in natural sunlight, surrounded by greenery or subtle environmental details like plants, leaves, or a wooden surface. The lighting feels bright and organic, giving a fresh, authentic vibe. The product remains clearly visible and central. No humans present.",
    "A clean, realistic product display photo where the product is placed in a simple, bright, nature-inspired setting — for example, a light-colored surface with plants or organic materials around it. The lighting is soft and natural, highlighting the product as the main subject. No humans included.",
    "A high-quality photo of the product positioned near a bright window or on a balcony ledge, with natural sunlight creating gentle shadows. The background shows hints of the outdoors — sky, plants, or buildings — softly blurred to keep focus on the product. No humans visible.",
    "A professional, realistic image of the product placed on a wooden or marble tabletop in a bright room. The background includes soft natural lighting and minimalist decor elements such as books, plants, or ceramics that enhance the lifestyle feel. The product is the main focus. No people in the frame.",
]

_last_prompt = None

def get_random_prompt():
    global _last_prompt
    available = [p for p in prompts_image_ex if p != _last_prompt]
    choice = random.choice(available)
    _last_prompt = choice
    return choice