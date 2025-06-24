import json
import boto3
import logging
from datetime import datetime
from typing import Dict, Any, List
import os

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb')
bedrock_client = boto3.client('bedrock-runtime')

# Environment variables
USERS_TABLE = os.environ['USERS_TABLE']
LEARNING_PATHS_TABLE = os.environ['LEARNING_PATHS_TABLE']

# DynamoDB tables
users_table = dynamodb.Table(USERS_TABLE)
learning_paths_table = dynamodb.Table(LEARNING_PATHS_TABLE)

def handler(event, context):
    """
    AI Career Coach handler - provides personalized career guidance
    """
    try:
        logger.info(f"Career coach request: {json.dumps(event)}")
        
        # Parse request
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', {})
        
        user_id = body.get('userId')
        message = body.get('message', '')
        conversation_id = body.get('conversationId')
        context_data = body.get('context', {})
        
        if not user_id or not message:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Missing required fields: userId, message'
                })
            }
        
        # Get user profile and context
        user_context = get_user_context(user_id)
        
        # Generate AI response
        ai_response = generate_career_advice(
            message=message,
            user_context=user_context,
            additional_context=context_data
        )
        
        # Store conversation
        conversation_record = store_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=message,
            ai_response=ai_response,
            context=context_data
        )
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'response': ai_response,
                'conversationId': conversation_record['conversationId'],
                'timestamp': conversation_record['timestamp']
            })
        }
        
    except Exception as e:
        logger.error(f"Error in career coach: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }

def get_user_context(user_id: str) -> Dict[str, Any]:
    """Retrieve user context from various sources"""
    try:
        # Get user profile
        user_response = users_table.get_item(Key={'userId': user_id})
        user_data = user_response.get('Item', {})
        
        # Get learning paths
        learning_response = learning_paths_table.query(
            KeyConditionExpression='userId = :uid',
            ExpressionAttributeValues={':uid': user_id},
            Limit=5,
            ScanIndexForward=False  # Get most recent first
        )
        learning_paths = learning_response.get('Items', [])
        
        return {
            'profile': user_data,
            'learningPaths': learning_paths,
            'careerGoals': user_data.get('careerGoals', []),
            'currentRole': user_data.get('currentRole', ''),
            'experience': user_data.get('experienceLevel', ''),
            'skills': user_data.get('skills', []),
            'interests': user_data.get('interests', [])
        }
        
    except Exception as e:
        logger.error(f"Error getting user context: {str(e)}")
        return {}

def generate_career_advice(message: str, user_context: Dict, additional_context: Dict) -> Dict[str, Any]:
    """Generate personalized career advice using Amazon Bedrock"""
    try:
        # Build context-aware prompt
        prompt = build_career_coaching_prompt(message, user_context, additional_context)
        
        # Call Bedrock
        body = json.dumps({
            "prompt": prompt,
            "max_tokens": 1500,
            "temperature": 0.7,
            "top_p": 0.9
        })
        
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-v2",
            body=body,
            contentType="application/json"
        )
        
        response_body = json.loads(response['body'].read())
        ai_text = response_body.get('completion', '')
        
        # Parse and structure the response
        structured_response = parse_career_advice(ai_text)
        
        # Add personalized recommendations
        structured_response['personalizedActions'] = generate_action_items(
            message, user_context, structured_response
        )
        
        return structured_response
        
    except Exception as e:
        logger.error(f"Error generating career advice: {str(e)}")
        return {
            'advice': "I'm having trouble processing your request right now. Please try again in a moment.",
            'category': 'general',
            'confidence': 0.5,
            'actionItems': ['Please try your question again'],
            'resources': []
        }

def build_career_coaching_prompt(message: str, user_context: Dict, additional_context: Dict) -> str:
    """Build a comprehensive prompt for career coaching"""
    
    profile = user_context.get('profile', {})
    current_role = profile.get('currentRole', 'Not specified')
    career_goals = profile.get('careerGoals', [])
    experience = profile.get('experienceLevel', 'Not specified')
    skills = profile.get('skills', [])
    
    prompt = f"""
You are an expert AI career coach with deep knowledge of technology careers, industry trends, and professional development. You provide personalized, actionable advice.

USER CONTEXT:
- Current Role: {current_role}
- Experience Level: {experience}
- Career Goals: {', '.join(career_goals) if career_goals else 'Not specified'}
- Key Skills: {', '.join(skills[:10]) if skills else 'Not specified'}
- Recent Learning Paths: {len(user_context.get('learningPaths', []))} active paths

USER QUESTION: {message}

ADDITIONAL CONTEXT: {json.dumps(additional_context) if additional_context else 'None'}

Please provide comprehensive career advice that includes:

1. DIRECT ADVICE: Clear, actionable guidance addressing their specific question
2. CATEGORY: Classify the advice type (career_transition, skill_development, interview_prep, salary_negotiation, networking, leadership, etc.)
3. CONFIDENCE: Your confidence level in this advice (0.0-1.0)
4. ACTION ITEMS: 3-5 specific, actionable steps they can take immediately
5. RESOURCES: Relevant learning resources, tools, or platforms
6. TIMELINE: Suggested timeframe for implementing advice
7. SUCCESS METRICS: How they can measure progress

Format your response as a structured analysis that's both comprehensive and practical. Be encouraging but realistic. Consider current market conditions and industry trends.

Focus on being:
- Specific and actionable
- Tailored to their experience level
- Realistic about timelines
- Encouraging but honest about challenges
- Up-to-date with industry trends
"""
    
    return prompt

def parse_career_advice(ai_text: str) -> Dict[str, Any]:
    """Parse and structure the AI career advice response"""
    
    # Default structure
    structured_response = {
        'advice': ai_text,
        'category': 'general',
        'confidence': 0.8,
        'actionItems': [],
        'resources': [],
        'timeline': 'Ongoing',
        'successMetrics': []
    }
    
    try:
        # Try to extract structured information from the response
        lines = ai_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Identify sections
            if 'DIRECT ADVICE:' in line.upper():
                current_section = 'advice'
                structured_response['advice'] = line.split(':', 1)[1].strip()
            elif 'CATEGORY:' in line.upper():
                current_section = 'category'
                structured_response['category'] = line.split(':', 1)[1].strip().lower()
            elif 'CONFIDENCE:' in line.upper():
                current_section = 'confidence'
                try:
                    conf_text = line.split(':', 1)[1].strip()
                    structured_response['confidence'] = float(conf_text.split()[0])
                except:
                    structured_response['confidence'] = 0.8
            elif 'ACTION ITEMS:' in line.upper():
                current_section = 'actionItems'
            elif 'RESOURCES:' in line.upper():
                current_section = 'resources'
            elif 'TIMELINE:' in line.upper():
                current_section = 'timeline'
                structured_response['timeline'] = line.split(':', 1)[1].strip()
            elif 'SUCCESS METRICS:' in line.upper():
                current_section = 'successMetrics'
            elif line.startswith(('-', '•', '1.', '2.', '3.', '4.', '5.')):
                # This is a list item
                item = line.lstrip('-•123456789. ').strip()
                if current_section == 'actionItems':
                    structured_response['actionItems'].append(item)
                elif current_section == 'resources':
                    structured_response['resources'].append(item)
                elif current_section == 'successMetrics':
                    structured_response['successMetrics'].append(item)
            elif current_section == 'advice' and line:
                # Continue building the advice text
                structured_response['advice'] += ' ' + line
        
        # Ensure we have at least some action items
        if not structured_response['actionItems']:
            structured_response['actionItems'] = [
                'Reflect on the advice provided',
                'Create a specific action plan',
                'Set measurable goals',
                'Track your progress regularly'
            ]
        
        return structured_response
        
    except Exception as e:
        logger.error(f"Error parsing career advice: {str(e)}")
        return structured_response

def generate_action_items(message: str, user_context: Dict, advice_response: Dict) -> List[Dict[str, Any]]:
    """Generate personalized action items based on user context and advice"""
    
    action_items = []
    category = advice_response.get('category', 'general')
    
    # Category-specific action items
    if category == 'skill_development':
        action_items.extend([
            {
                'action': 'Identify skill gaps',
                'description': 'Compare your current skills with job requirements',
                'priority': 'high',
                'timeframe': '1 week'
            },
            {
                'action': 'Create learning plan',
                'description': 'Choose courses or resources for skill development',
                'priority': 'high',
                'timeframe': '2 weeks'
            }
        ])
    elif category == 'career_transition':
        action_items.extend([
            {
                'action': 'Research target roles',
                'description': 'Study job descriptions and requirements',
                'priority': 'high',
                'timeframe': '1 week'
            },
            {
                'action': 'Network with professionals',
                'description': 'Connect with people in your target field',
                'priority': 'medium',
                'timeframe': '2-4 weeks'
            }
        ])
    elif category == 'interview_prep':
        action_items.extend([
            {
                'action': 'Practice common questions',
                'description': 'Prepare answers for behavioral and technical questions',
                'priority': 'high',
                'timeframe': '1 week'
            },
            {
                'action': 'Mock interviews',
                'description': 'Practice with friends or use online platforms',
                'priority': 'high',
                'timeframe': '2 weeks'
            }
        ])
    
    # Add general action items if none were generated
    if not action_items:
        action_items = [
            {
                'action': 'Set specific goals',
                'description': 'Define clear, measurable objectives',
                'priority': 'high',
                'timeframe': '1 week'
            },
            {
                'action': 'Create action plan',
                'description': 'Break down goals into actionable steps',
                'priority': 'medium',
                'timeframe': '2 weeks'
            }
        ]
    
    return action_items[:5]  # Limit to 5 action items

def store_conversation(user_id: str, conversation_id: str, user_message: str, 
                      ai_response: Dict, context: Dict) -> Dict[str, Any]:
    """Store conversation in DynamoDB for future reference"""
    
    import uuid
    
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    timestamp = int(datetime.utcnow().timestamp())
    
    conversation_record = {
        'userId': user_id,
        'conversationId': conversation_id,
        'timestamp': timestamp,
        'userMessage': user_message,
        'aiResponse': ai_response,
        'context': context,
        'category': ai_response.get('category', 'general'),
        'confidence': ai_response.get('confidence', 0.8)
    }
    
    try:
        # Store in a conversations table (would need to be created)
        # For now, we'll store in the learning paths table with a different structure
        learning_paths_table.put_item(
            Item={
                'userId': user_id,
                'pathId': f"conversation_{conversation_id}",
                'type': 'conversation',
                'timestamp': timestamp,
                'data': conversation_record
            }
        )
        
        logger.info(f"Stored conversation {conversation_id} for user {user_id}")
        
    except Exception as e:
        logger.error(f"Error storing conversation: {str(e)}")
    
    return conversation_record

def get_conversation_history(user_id: str, limit: int = 10) -> List[Dict]:
    """Retrieve recent conversation history for context"""
    try:
        response = learning_paths_table.query(
            KeyConditionExpression='userId = :uid',
            FilterExpression='#type = :type',
            ExpressionAttributeNames={'#type': 'type'},
            ExpressionAttributeValues={
                ':uid': user_id,
                ':type': 'conversation'
            },
            Limit=limit,
            ScanIndexForward=False
        )
        
        conversations = []
        for item in response.get('Items', []):
            conversations.append(item.get('data', {}))
        
        return conversations
        
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {str(e)}")
        return []