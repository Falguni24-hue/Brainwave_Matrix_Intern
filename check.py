import requests

def analyze_facebook_posts(user_id, access_token):
    url = f"https://graph.facebook.com/v15.0/{8539716629464572}/posts"
    params = {
        'access_token': "EAAJeKDEfAuMBO3nPOfsDPY2iu1BdPxYuYZB3HpZCfGoHkVZC8vIApnzKHnuGg1mS8vylg5SlbY6b2EvBp6Jd74fFDwmQ9bWwd8xwxGHvGKMk7o3JkJ22JZCVGkHiyAxgkW8cxdQIqntDTAlDEiBZAU1uqJZC1fcUZC3ccAqegmkOUaSJcZCEPClTipQU6mMbzbeM6GezERZAMLTNsNkZA0uaQZB9Ax524qQZAWZBjs1cmZAePAAFJJr1KehOAk1QZDZD",
        'fields': 'message',
        'limit': 100  # number of posts you want to fetch
    }
    response = requests.get(url, params=params)
    data = response.json()
    posts = []
    
    for post in data['data']:
        posts.append({'text': post.get('message', ''), 'sentiment': 0})  # Replace sentiment after analysis
        
    return posts
