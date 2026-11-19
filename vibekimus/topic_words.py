"""Words a topic of pictures may be named after (Explore). A topic gets the two words
whose text embeddings are closest to its centre, so the list covers what personal
pictures commonly show, in short phrases the models understand."""

TOPIC_WORDS = (
    # people
    "people", "a selfie", "a portrait", "a group photo", "a baby", "children", "a wedding", "a party",
    "a concert", "sports", "a crowd", "a family", "friends", "a couple", "an old photo",
    # animals
    "a cat", "a dog", "a bird", "a horse", "fish", "insects", "wild animals", "a pet",
    # nature and places
    "mountains", "a beach", "the sea", "a forest", "flowers", "a sunset", "snow", "the sky", "a lake",
    "a garden", "a field", "a desert", "a waterfall", "the night sky", "clouds", "trees",
    "a city", "a street", "buildings", "a city at night", "a church", "a castle", "a bridge", "a house",
    "a room", "a kitchen", "an office", "a shop", "a restaurant", "a park", "a road trip", "a holiday",
    # vehicles
    "a car", "a motorcycle", "a bicycle", "a train", "an airplane", "a boat", "a truck",
    # food
    "food", "a meal", "a dessert", "drinks", "coffee", "beer", "fruit", "pizza", "cooking",
    # text and screens
    "a document", "a receipt", "a page of text", "handwriting", "a book", "a presentation slide", "a chart",
    "a table of numbers", "a form", "a ticket", "a map", "a screenshot", "a chat conversation", "a website",
    "a video game", "a computer desktop", "program code", "a phone screen", "an error message", "a social media post",
    "a video call",
    # pictures that are not photos
    "anime", "a cartoon", "a drawing", "a painting", "a meme", "a comic", "a 3D render", "a logo", "an icon",
    "a poster", "pixel art", "a sketch", "a wallpaper", "a diagram", "text on a plain background",
    # things
    "clothes", "shoes", "furniture", "electronics", "a computer", "a phone", "toys", "tools", "weapons",
    "jewellery", "a product photo", "money", "cosmetics", "a car interior", "a construction site",
    # moments
    "Christmas", "a birthday", "fireworks", "a museum", "an aquarium", "a zoo", "a stadium", "a gym",
    "a swimming pool", "a camping trip", "a hospital", "a classroom",
)
