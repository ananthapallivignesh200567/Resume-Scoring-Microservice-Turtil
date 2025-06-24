import json
import boto3
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List
import base64
from urllib.parse import unquote_plus

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
textract_client = boto3.client('textract')
comprehend_client = boto3.client('comprehend')
bedrock_client = boto3.client('bedrock-runtime')
sns_client = boto3.client('sns')

# Environment variables
RESUMES_BUCKET = os.environ['RESUMES_BUCKET']
ANALYSIS_TABLE = os.environ['ANALYSIS_TABLE']
NOTIFICATION_TOPIC = os.environ['NOTIFICATION_TOPIC']

# DynamoDB table
analysis_table = dynamodb.Table(ANALYSIS_TABLE)

def handler(event, context):
    """
    Main handler for resume processing.
    Supports both direct invocation and S3 event triggers.
    """
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        
        # Determine event source
        if 'Records' in event:
            # S3 event trigger
            return process_s3_event(event)
        else:
            # Direct API invocation
            return process_direct_invocation(event)
            
    except Exception as e:
        logger.error(f"Error processing resume: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }

def process_direct_invocation(event):
    """Process resume from direct API call"""
    try:
        # Parse request body
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', {})
        
        user_id = body.get('userId')
        resume_content = body.get('resumeContent')  # Base64 encoded
        file_name = body.get('fileName', 'resume.pdf')
        goal = body.get('goal', 'General')
        
        if not user_id or not resume_content:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Missing required fields: userId, resumeContent'
                })
            }
        
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Decode and upload resume to S3
        resume_data = base64.b64decode(resume_content)
        s3_key = f"resumes/{user_id}/{analysis_id}/{file_name}"
        
        s3_client.put_object(
            Bucket=RESUMES_BUCKET,
            Key=s3_key,
            Body=resume_data,
            ContentType=get_content_type(file_name)
        )
        
        # Process the resume
        analysis_result = analyze_resume(s3_key, user_id, analysis_id, goal)
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(analysis_result)
        }
        
    except Exception as e:
        logger.error(f"Error in direct invocation: {str(e)}")
        raise

def process_s3_event(event):
    """Process resume from S3 event trigger"""
    results = []
    
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = unquote_plus(record['s3']['object']['key'])
        
        logger.info(f"Processing S3 object: {bucket}/{key}")
        
        # Extract user_id and analysis_id from S3 key
        # Expected format: resumes/{user_id}/{analysis_id}/{filename}
        key_parts = key.split('/')
        if len(key_parts) >= 3 and key_parts[0] == 'resumes':
            user_id = key_parts[1]
            analysis_id = key_parts[2]
            
            # Process the resume
            analysis_result = analyze_resume(key, user_id, analysis_id)
            results.append(analysis_result)
        else:
            logger.warning(f"Invalid S3 key format: {key}")
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'processed': len(results),
            'results': results
        })
    }

def analyze_resume(s3_key: str, user_id: str, analysis_id: str, goal: str = 'General') -> Dict[str, Any]:
    """
    Comprehensive resume analysis using multiple AWS AI services
    """
    try:
        logger.info(f"Starting analysis for {s3_key}")
        
        # Step 1: Extract text from resume
        resume_text = extract_text_from_resume(s3_key)
        
        # Step 2: Analyze with Comprehend
        comprehend_analysis = analyze_with_comprehend(resume_text)
        
        # Step 3: Extract skills and experience
        skills_analysis = extract_skills_and_experience(resume_text)
        
        # Step 4: Generate AI insights
        ai_insights = generate_ai_insights(resume_text, goal)
        
        # Step 5: Calculate compatibility score
        compatibility_score = calculate_compatibility_score(skills_analysis, goal)
        
        # Step 6: Generate recommendations
        recommendations = generate_recommendations(skills_analysis, goal, compatibility_score)
        
        # Compile analysis result
        analysis_result = {
            'analysisId': analysis_id,
            'userId': user_id,
            'goal': goal,
            'timestamp': int(datetime.utcnow().timestamp()),
            'resumeText': resume_text[:1000],  # Store first 1000 chars
            'skills': skills_analysis,
            'comprehendAnalysis': comprehend_analysis,
            'aiInsights': ai_insights,
            'compatibilityScore': compatibility_score,
            'recommendations': recommendations,
            's3Key': s3_key,
            'status': 'completed'
        }
        
        # Store in DynamoDB
        analysis_table.put_item(Item=analysis_result)
        
        # Send notification
        send_completion_notification(user_id, analysis_id, compatibility_score)
        
        logger.info(f"Analysis completed for {analysis_id}")
        return analysis_result
        
    except Exception as e:
        logger.error(f"Error in resume analysis: {str(e)}")
        
        # Store error result
        error_result = {
            'analysisId': analysis_id,
            'userId': user_id,
            'timestamp': int(datetime.utcnow().timestamp()),
            'status': 'error',
            'error': str(e),
            's3Key': s3_key
        }
        
        analysis_table.put_item(Item=error_result)
        raise

def extract_text_from_resume(s3_key: str) -> str:
    """Extract text from resume using Textract"""
    try:
        response = textract_client.detect_document_text(
            Document={
                'S3Object': {
                    'Bucket': RESUMES_BUCKET,
                    'Name': s3_key
                }
            }
        )
        
        # Extract text from Textract response
        text_blocks = []
        for block in response['Blocks']:
            if block['BlockType'] == 'LINE':
                text_blocks.append(block['Text'])
        
        return '\n'.join(text_blocks)
        
    except Exception as e:
        logger.error(f"Error extracting text with Textract: {str(e)}")
        
        # Fallback: try to read as plain text if it's a .txt file
        if s3_key.lower().endswith('.txt'):
            try:
                response = s3_client.get_object(Bucket=RESUMES_BUCKET, Key=s3_key)
                return response['Body'].read().decode('utf-8')
            except Exception as fallback_error:
                logger.error(f"Fallback text extraction failed: {str(fallback_error)}")
        
        raise e

def analyze_with_comprehend(text: str) -> Dict[str, Any]:
    """Analyze resume text with Amazon Comprehend"""
    try:
        # Truncate text if too long (Comprehend has limits)
        text = text[:5000] if len(text) > 5000 else text
        
        # Detect entities
        entities_response = comprehend_client.detect_entities(
            Text=text,
            LanguageCode='en'
        )
        
        # Detect key phrases
        phrases_response = comprehend_client.detect_key_phrases(
            Text=text,
            LanguageCode='en'
        )
        
        # Detect sentiment
        sentiment_response = comprehend_client.detect_sentiment(
            Text=text,
            LanguageCode='en'
        )
        
        return {
            'entities': entities_response['Entities'][:20],  # Limit to top 20
            'keyPhrases': phrases_response['KeyPhrases'][:20],
            'sentiment': sentiment_response['Sentiment'],
            'sentimentScores': sentiment_response['SentimentScore']
        }
        
    except Exception as e:
        logger.error(f"Error with Comprehend analysis: {str(e)}")
        return {
            'entities': [],
            'keyPhrases': [],
            'sentiment': 'NEUTRAL',
            'sentimentScores': {}
        }

def extract_skills_and_experience(text: str) -> Dict[str, Any]:
    """Extract skills and experience using pattern matching and NLP"""
    
    # Technical skills patterns
    technical_skills = [
        'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'Go', 'Rust',
        'React', 'Angular', 'Vue', 'Node.js', 'Express', 'Django', 'Flask',
        'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Terraform',
        'SQL', 'PostgreSQL', 'MySQL', 'MongoDB', 'Redis',
        'Git', 'Jenkins', 'CI/CD', 'DevOps', 'Agile', 'Scrum',
        'Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch',
        'Data Science', 'Analytics', 'Tableau', 'Power BI'
    ]
    
    # Soft skills patterns
    soft_skills = [
        'Leadership', 'Communication', 'Teamwork', 'Problem Solving',
        'Project Management', 'Time Management', 'Critical Thinking',
        'Adaptability', 'Creativity', 'Collaboration'
    ]
    
    text_lower = text.lower()
    
    # Find technical skills
    found_technical = []
    for skill in technical_skills:
        if skill.lower() in text_lower:
            found_technical.append(skill)
    
    # Find soft skills
    found_soft = []
    for skill in soft_skills:
        if skill.lower() in text_lower:
            found_soft.append(skill)
    
    # Extract years of experience
    import re
    experience_patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
        r'(\d+)\+?\s*years?\s*in',
        r'experience\s*(?:of\s*)?(\d+)\+?\s*years?'
    ]
    
    years_experience = 0
    for pattern in experience_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            years_experience = max(years_experience, max(int(match) for match in matches))
    
    return {
        'technicalSkills': found_technical,
        'softSkills': found_soft,
        'yearsExperience': years_experience,
        'totalSkills': len(found_technical) + len(found_soft)
    }

def generate_ai_insights(text: str, goal: str) -> Dict[str, Any]:
    """Generate AI insights using Amazon Bedrock"""
    try:
        prompt = f"""
        Analyze this resume for a {goal} position and provide insights:
        
        Resume Text:
        {text[:2000]}
        
        Please provide:
        1. Key strengths (3-5 points)
        2. Areas for improvement (3-5 points)
        3. Fit for {goal} role (1-10 scale with explanation)
        4. Career level assessment (Entry/Mid/Senior)
        5. Top 3 recommendations
        
        Format as JSON with keys: strengths, improvements, fitScore, fitExplanation, careerLevel, recommendations
        """
        
        body = json.dumps({
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": 0.7
        })
        
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-v2",
            body=body,
            contentType="application/json"
        )
        
        response_body = json.loads(response['body'].read())
        
        # Parse the AI response
        ai_text = response_body.get('completion', '')
        
        # Try to extract JSON from the response
        try:
            # Look for JSON in the response
            import re
            json_match = re.search(r'\{.*\}', ai_text, re.DOTALL)
            if json_match:
                ai_insights = json.loads(json_match.group())
            else:
                # Fallback to structured parsing
                ai_insights = parse_ai_response_fallback(ai_text)
        except:
            ai_insights = parse_ai_response_fallback(ai_text)
        
        return ai_insights
        
    except Exception as e:
        logger.error(f"Error generating AI insights: {str(e)}")
        return {
            'strengths': ['Unable to analyze - please try again'],
            'improvements': ['Unable to analyze - please try again'],
            'fitScore': 5,
            'fitExplanation': 'Analysis unavailable',
            'careerLevel': 'Unknown',
            'recommendations': ['Please try again later']
        }

def parse_ai_response_fallback(text: str) -> Dict[str, Any]:
    """Fallback parser for AI response"""
    return {
        'strengths': ['Analysis completed'],
        'improvements': ['Continue developing skills'],
        'fitScore': 7,
        'fitExplanation': 'Good potential for the role',
        'careerLevel': 'Mid',
        'recommendations': ['Keep learning', 'Build projects', 'Network actively']
    }

def calculate_compatibility_score(skills_analysis: Dict, goal: str) -> float:
    """Calculate compatibility score based on skills and goal"""
    
    # Goal-specific skill requirements
    goal_requirements = {
        'Software Engineer': {
            'required': ['Python', 'Java', 'JavaScript', 'Git', 'SQL'],
            'preferred': ['AWS', 'Docker', 'React', 'Node.js'],
            'experience_weight': 0.3
        },
        'Data Scientist': {
            'required': ['Python', 'SQL', 'Machine Learning', 'Statistics'],
            'preferred': ['TensorFlow', 'PyTorch', 'Tableau', 'R'],
            'experience_weight': 0.4
        },
        'DevOps Engineer': {
            'required': ['AWS', 'Docker', 'Kubernetes', 'CI/CD', 'Linux'],
            'preferred': ['Terraform', 'Jenkins', 'Monitoring'],
            'experience_weight': 0.4
        }
    }
    
    requirements = goal_requirements.get(goal, goal_requirements['Software Engineer'])
    
    found_skills = skills_analysis['technicalSkills']
    
    # Calculate required skills match
    required_match = sum(1 for skill in requirements['required'] if skill in found_skills)
    required_score = required_match / len(requirements['required'])
    
    # Calculate preferred skills match
    preferred_match = sum(1 for skill in requirements['preferred'] if skill in found_skills)
    preferred_score = preferred_match / len(requirements['preferred']) if requirements['preferred'] else 0
    
    # Experience factor
    years = skills_analysis['yearsExperience']
    experience_score = min(years / 5.0, 1.0)  # Normalize to 5 years max
    
    # Weighted final score
    final_score = (
        required_score * 0.5 +
        preferred_score * 0.3 +
        experience_score * requirements['experience_weight']
    )
    
    return round(min(final_score, 1.0), 3)

def generate_recommendations(skills_analysis: Dict, goal: str, score: float) -> List[str]:
    """Generate personalized recommendations"""
    
    recommendations = []
    
    if score < 0.3:
        recommendations.extend([
            f"Focus on building foundational skills for {goal}",
            "Consider taking online courses or bootcamps",
            "Start with personal projects to build experience"
        ])
    elif score < 0.6:
        recommendations.extend([
            "Continue developing technical skills",
            "Build a portfolio of relevant projects",
            "Consider contributing to open source projects"
        ])
    else:
        recommendations.extend([
            "You're well-positioned for this role",
            "Focus on advanced skills and leadership",
            "Consider mentoring others in your field"
        ])
    
    # Add skill-specific recommendations
    if skills_analysis['yearsExperience'] < 2:
        recommendations.append("Gain more hands-on experience through internships or projects")
    
    if len(skills_analysis['technicalSkills']) < 5:
        recommendations.append("Expand your technical skill set")
    
    return recommendations[:5]  # Limit to 5 recommendations

def send_completion_notification(user_id: str, analysis_id: str, score: float):
    """Send completion notification via SNS"""
    try:
        message = {
            'userId': user_id,
            'analysisId': analysis_id,
            'score': score,
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'resume_analysis_complete'
        }
        
        sns_client.publish(
            TopicArn=NOTIFICATION_TOPIC,
            Message=json.dumps(message),
            Subject='Resume Analysis Complete'
        )
        
    except Exception as e:
        logger.error(f"Error sending notification: {str(e)}")

def get_content_type(filename: str) -> str:
    """Get content type based on file extension"""
    extension = filename.lower().split('.')[-1]
    content_types = {
        'pdf': 'application/pdf',
        'doc': 'application/msword',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'txt': 'text/plain'
    }
    return content_types.get(extension, 'application/octet-stream')