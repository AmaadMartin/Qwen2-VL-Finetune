from PIL import Image
import json

def create_text_to_point_conversation_from_datapoint(image, task, coordinates, K):
    # 2     |      1
    #       |  
    # --------------  
    #       |
    # 3     |      4
    conversation = []
    x, y = coordinates
    quadrant = 0
    images = [image]
    for _ in range(K):
        if x >= 0.5 and y <= 0.5:
            x, y = ((x - 0.5) * 2, y * 2)
            quadrant = 1
            images.append(images[-1].crop((images[-1].width // 2, 0, images[-1].width, images[-1].height // 2)))
        elif x <= 0.5 and y <= 0.5:
            x, y = (x * 2, y * 2)
            quadrant = 2
            images.append(images[-1].crop((0, 0, images[-1].width // 2, images[-1].height // 2)))
        elif x <= 0.5 and y >= 0.5:
            x, y = (x * 2, (y - 0.5) * 2)
            quadrant = 3
            images.append(images[-1].crop((0, images[-1].height // 2, images[-1].width // 2, images[-1].height)))
        elif x >= 0.5 and y >= 0.5:
            x, y = ((x - 0.5) * 2, (y - 0.5) * 2)
            quadrant = 4
            images.append(images[-1].crop((images[-1].width // 2, images[-1].height // 2, images[-1].width, images[-1].height)))
        else:
            raise ValueError(f"Invalid coordinates: {coordinates}")
            
        conversation.append({
            "from": "human",
            "value": f"<image>\nIn this UI screenshot, what is the partition of the element corresponding to the command \"{task}\" (with quadrant number)?",
        })
        conversation.append({
            "from": "gpt",
            "value": f"{quadrant}"
        })  

    # add the last image to the conversation
    conversation.append({
        "from": "human",
        "value": f"<image>\nIn this UI screenshot, what is the position of the element corresponding to the command \"{task}\" (with point)?"
    })
    conversation.append({
        "from": "gpt",
        "value": f"({x}, {y})"
    })

    sources = {
        "image": images,
        "conversations": conversation
    }

    return sources

def create_text_to_bbox_conversation_from_datapoint(image, task, bbox):
    images = [image]
    conversation = [
        {
            "from": "human",
            "value": f"<image>\nIn this UI screenshot, what is the position of the element corresponding to the command \"{task}\" (with bbox)?",
        },
        {
            "from": "gpt",
            "value": f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})"
        }
    ]
    sources = {
        "image": images,
        "conversations": conversation
    }
    return sources

def create_point_to_text_conversation_from_datapoint(image, element, coordinates):
    images = [image]
    conversation = [
        {
            "from": "human",
            "value": f"<image>\nIn this UI screenshot, what is the element corresponding to the point ({coordinates[0]}, {coordinates[1]}) on the screen?",
        },
        {
            "from": "gpt",
            "value": f"{element}"
        }
    ]
    sources = {
        "image": images,
        "conversations": conversation
    }
    return sources

def create_bbox_to_text_conversation_from_datapoint(image, element, bbox):
    images = [image]
    conversation = [
        {
            "from": "human",
            "value": f"<image>\nIn this UI screenshot, what is the element corresponding to the bounding box ({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}) on the screen?",
        },
        {
            "from": "gpt",
            "value": f"{element}"
        }
    ]
    sources = {
        "image": images,
        "conversations": conversation
    }
    return sources

def create_ui_summarization_conversation_from_datapoint(image, description):
    images = [image]
    conversation = [
        {
            "from": "human", 
            "value": f"<image>\nCan you provide a detailed description of the interface screenshot shown?",
        },
        {
            "from": "gpt",
            "value": f"{description}"
        }
    ]
    sources = {
        "image": images,
        "conversations": conversation
    }
    return sources

def create_widget_captioning_conversation_from_datapoint(image, coordinates, caption):
    images = [image]
    conversation = [
        {
            "from": "human",
            "value": f"<image>\nFrom this UI screenshot, can you caption the widget corresponding to the point \"{coordinates}\" on the screen?",
        },
        {
            "from": "gpt",
            "value": f"{caption}"
        }
    ]
    sources = {
        "image": images,
        "conversations": conversation
    }
    return sources

def create_general_conversation_from_datapoint(images, conversations):
    role_mapping = {"user": "human", "assistant": "gpt"}
    new_conversation = []
    for conversation in conversations:
        content = conversation["content"]
        text = ""
        for item in content:
            if item["type"] == "image":
                text += f"<image>\n"
            elif item["type"] == "text":
                text += item["text"]
        new_conversation.append({
            "from": role_mapping[conversation["role"]],
            "value": text
        })
    sources = {
        "image": images,
        "conversations": new_conversation
    }
    return sources

def convert_sources(sources, K):
    # images, text, coordinates, bbox, data_type, conversation = None
    images = sources.get("images", None)
    images = [Image.open(image) for image in images]
    text = sources.get("text", None)
    coordinates = sources.get("coordinates", None)
    bbox = sources.get("bbox", None)
    data_type = sources.get("type", None)
    conversation = sources.get("conversation", None)
    if data_type == "text_to_point":
        return create_text_to_point_conversation_from_datapoint(images[0], text, coordinates, K)
    elif data_type == "text_to_bbox":
        return create_text_to_bbox_conversation_from_datapoint(images[0], text, bbox)
    elif data_type == "point_to_text":
        return create_point_to_text_conversation_from_datapoint(images[0], text, coordinates)
    elif data_type == "bbox_to_text":
        return create_bbox_to_text_conversation_from_datapoint(images[0], text, bbox)
    elif data_type == "ui_summarization":
        return create_ui_summarization_conversation_from_datapoint(images[0], text)
    elif data_type == "widget_captioning":
        return create_widget_captioning_conversation_from_datapoint(images[0], coordinates, text)
    elif data_type == "general_llava":
        return create_general_conversation_from_datapoint(images, conversation)
    else:
        raise ValueError(f"Invalid type: {data_type}")