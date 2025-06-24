import json
import boto3
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import os
import time
from bs4 import BeautifulSoup
import re

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb')
bedrock_client = boto3.client('bedrock-runtime')

# Environment variables
JOB_MARKET_TABLE = os.environ['JOB_MARKET_TABLE']

# DynamoDB table
job_market_table = dynamodb.Table(JOB_MARKET_TABLE)

def handler(event, context):
    """
    Job Market Intelligence handler - collects and analyzes job market data
    """
    try:
        logger.info(f"Job market request: {json.dumps(event)}")
        
        # Determine the operation type
        if 'source' in event and event['source'] == 'aws.events':
            # Scheduled event - collect market data
            return collect_market_data()
        else:
            # API request - return market insights
            return get_market_insights(event)
            
    except Exception as e:
        logger.error(f"Error in job market handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }

def collect_market_data():
    """Collect job market data from various sources"""
    try:
        logger.info("Starting job market data collection")
        
        # Collect data from multiple sources
        results = {
            'timestamp': int(datetime.utcnow().timestamp()),
            'sources': {}
        }
        
        # 1. Collect from job boards (simulated - in real implementation, use APIs)
        job_board_data = collect_job_board_data()
        results['sources']['job_boards'] = job_board_data
        
        # 2. Collect salary data
        salary_data = collect_salary_data()
        results['sources']['salary_data'] = salary_data
        
        # 3. Collect skill trends
        skill_trends = collect_skill_trends()
        results['sources']['skill_trends'] = skill_trends
        
        # 4. Generate market insights using AI
        market_insights = generate_market_insights(results)
        results['insights'] = market_insights
        
        # Store in DynamoDB
        store_market_data(results)
        
        logger.info("Job market data collection completed")
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Market data collection completed',
                'timestamp': results['timestamp'],
                'sources_collected': len(results['sources'])
            })
        }
        
    except Exception as e:
        logger.error(f"Error collecting market data: {str(e)}")
        raise

def get_market_insights(event):
    """Return market insights based on query parameters"""
    try:
        # Parse query parameters
        query_params = event.get('queryStringParameters', {}) or {}
        role = query_params.get('role', 'Software Engineer')
        location = query_params.get('location', 'United States')
        experience_level = query_params.get('experience', 'mid')
        
        # Get recent market data
        market_data = get_recent_market_data()
        
        # Filter and analyze data for specific query
        insights = analyze_market_for_role(market_data, role, location, experience_level)
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(insights)
        }
        
    except Exception as e:
        logger.error(f"Error getting market insights: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Failed to get market insights',
                'message': str(e)
            })
        }

def collect_job_board_data() -> Dict[str, Any]:
    """Collect job postings data (simulated for demo)"""
    
    # In a real implementation, this would use APIs from:
    # - LinkedIn Jobs API
    # - Indeed API
    # - Glassdoor API
    # - AngelList API
    # - Company career pages
    
    # Simulated data structure
    job_data = {
        'total_jobs': 15420,
        'by_role': {
            'Software Engineer': 4200,
            'Data Scientist': 2100,
            'DevOps Engineer': 1800,
            'Product Manager': 1500,
            'Frontend Developer': 2000,
            'Backend Developer': 1900,
            'Full Stack Developer': 1920
        },
        'by_experience': {
            'entry': 3500,
            'mid': 8200,
            'senior': 3720
        },
        'by_location': {
            'San Francisco': 2800,
            'New York': 2400,
            'Seattle': 1900,
            'Austin': 1200,
            'Remote': 4500,
            'Other': 2620
        },
        'trending_companies': [
            'Google', 'Microsoft', 'Amazon', 'Meta', 'Apple',
            'Netflix', 'Uber', 'Airbnb', 'Stripe', 'OpenAI'
        ],
        'growth_rate': 0.15,  # 15% growth from last month
        'collection_timestamp': datetime.utcnow().isoformat()
    }
    
    return job_data

def collect_salary_data() -> Dict[str, Any]:
    """Collect salary information (simulated for demo)"""
    
    # In real implementation, use APIs from:
    # - Glassdoor API
    # - PayScale API
    # - Levels.fyi API
    # - Salary.com API
    
    salary_data = {
        'by_role': {
            'Software Engineer': {
                'entry': {'min': 85000, 'max': 120000, 'median': 102000},
                'mid': {'min': 120000, 'max': 180000, 'median': 150000},
                'senior': {'min': 180000, 'max': 300000, 'median': 220000}
            },
            'Data Scientist': {
                'entry': {'min': 90000, 'max': 130000, 'median': 110000},
                'mid': {'min': 130000, 'max': 200000, 'median': 165000},
                'senior': {'min': 200000, 'max': 350000, 'median': 250000}
            },
            'DevOps Engineer': {
                'entry': {'min': 80000, 'max': 115000, 'median': 97000},
                'mid': {'min': 115000, 'max': 170000, 'median': 142000},
                'senior': {'min': 170000, 'max': 280000, 'median': 210000}
            }
        },
        'by_location': {
            'San Francisco': {'multiplier': 1.4, 'cost_of_living': 1.6},
            'New York': {'multiplier': 1.3, 'cost_of_living': 1.5},
            'Seattle': {'multiplier': 1.2, 'cost_of_living': 1.3},
            'Austin': {'multiplier': 1.0, 'cost_of_living': 1.1},
            'Remote': {'multiplier': 0.9, 'cost_of_living': 1.0}
        },
        'trends': {
            'year_over_year_growth': 0.08,  # 8% salary growth
            'inflation_adjusted_growth': 0.03,  # 3% real growth
            'hot_skills_premium': {
                'AI/ML': 0.25,  # 25% premium
                'Cloud Architecture': 0.20,
                'Kubernetes': 0.18,
                'React': 0.12,
                'Python': 0.10
            }
        }
    }
    
    return salary_data

def collect_skill_trends() -> Dict[str, Any]:
    """Collect trending skills data"""
    
    # In real implementation, analyze:
    # - Job posting requirements
    # - GitHub trending repositories
    # - Stack Overflow surveys
    # - Course enrollment data
    # - Certification trends
    
    skill_trends = {
        'trending_up': [
            {'skill': 'Artificial Intelligence', 'growth': 0.45, 'demand_score': 9.2},
            {'skill': 'Machine Learning', 'growth': 0.38, 'demand_score': 9.0},
            {'skill': 'Kubernetes', 'growth': 0.35, 'demand_score': 8.5},
            {'skill': 'React', 'growth': 0.28, 'demand_score': 8.8},
            {'skill': 'Python', 'growth': 0.25, 'demand_score': 9.5},
            {'skill': 'AWS', 'growth': 0.22, 'demand_score': 9.1},
            {'skill': 'TypeScript', 'growth': 0.20, 'demand_score': 8.3},
            {'skill': 'Docker', 'growth': 0.18, 'demand_score': 8.7}
        ],
        'trending_down': [
            {'skill': 'jQuery', 'decline': -0.15, 'demand_score': 5.2},
            {'skill': 'PHP', 'decline': -0.08, 'demand_score': 6.8},
            {'skill': 'Ruby', 'decline': -0.05, 'demand_score': 6.5}
        ],
        'stable_high_demand': [
            {'skill': 'JavaScript', 'demand_score': 9.8},
            {'skill': 'Java', 'demand_score': 9.3},
            {'skill': 'SQL', 'demand_score': 9.4},
            {'skill': 'Git', 'demand_score': 9.6}
        ],
        'emerging_skills': [
            {'skill': 'Large Language Models', 'emergence_score': 8.9},
            {'skill': 'Prompt Engineering', 'emergence_score': 8.5},
            {'skill': 'Edge Computing', 'emergence_score': 7.8},
            {'skill': 'WebAssembly', 'emergence_score': 7.2}
        ]
    }
    
    return skill_trends

def generate_market_insights(market_data: Dict) -> Dict[str, Any]:
    """Generate AI-powered market insights"""
    try:
        prompt = f"""
        Analyze the following job market data and provide comprehensive insights:
        
        Job Market Data:
        {json.dumps(market_data, indent=2)}
        
        Please provide insights on:
        1. Overall market health and trends
        2. Most in-demand roles and skills
        3. Salary trends and predictions
        4. Geographic opportunities
        5. Advice for job seekers
        6. Predictions for next 6 months
        
        Format as structured JSON with keys: market_health, top_roles, salary_trends, 
        geographic_insights, job_seeker_advice, predictions
        """
        
        body = json.dumps({
            "prompt": prompt,
            "max_tokens": 2000,
            "temperature": 0.7
        })
        
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-v2",
            body=body,
            contentType="application/json"
        )
        
        response_body = json.loads(response['body'].read())
        ai_text = response_body.get('completion', '')
        
        # Try to parse JSON from response
        try:
            import re
            json_match = re.search(r'\{.*\}', ai_text, re.DOTALL)
            if json_match:
                insights = json.loads(json_match.group())
            else:
                insights = parse_insights_fallback(ai_text)
        except:
            insights = parse_insights_fallback(ai_text)
        
        return insights
        
    except Exception as e:
        logger.error(f"Error generating market insights: {str(e)}")
        return {
            'market_health': 'Strong',
            'top_roles': ['Software Engineer', 'Data Scientist'],
            'salary_trends': 'Upward',
            'geographic_insights': 'Remote work increasing',
            'job_seeker_advice': 'Focus on in-demand skills',
            'predictions': 'Continued growth in tech sector'
        }

def parse_insights_fallback(text: str) -> Dict[str, Any]:
    """Fallback parser for market insights"""
    return {
        'market_health': 'Strong - high demand for tech talent',
        'top_roles': [
            'Software Engineer',
            'Data Scientist', 
            'DevOps Engineer',
            'Product Manager'
        ],
        'salary_trends': 'Salaries continue to grow, especially for AI/ML skills',
        'geographic_insights': 'Remote work opportunities expanding globally',
        'job_seeker_advice': [
            'Focus on AI and machine learning skills',
            'Build cloud computing expertise',
            'Develop full-stack capabilities',
            'Contribute to open source projects'
        ],
        'predictions': 'AI and automation will drive job market evolution'
    }

def store_market_data(data: Dict):
    """Store market data in DynamoDB"""
    try:
        # Create a unique job ID for this data collection
        job_id = f"market_data_{data['timestamp']}"
        
        # Add TTL (30 days from now)
        ttl = int((datetime.utcnow() + timedelta(days=30)).timestamp())
        
        item = {
            'jobId': job_id,
            'scrapedAt': data['timestamp'],
            'ttl': ttl,
            'type': 'market_data',
            'data': data
        }
        
        job_market_table.put_item(Item=item)
        logger.info(f"Stored market data with ID: {job_id}")
        
    except Exception as e:
        logger.error(f"Error storing market data: {str(e)}")
        raise

def get_recent_market_data() -> Dict[str, Any]:
    """Get the most recent market data"""
    try:
        # Query for recent market data
        response = job_market_table.scan(
            FilterExpression='#type = :type',
            ExpressionAttributeNames={'#type': 'type'},
            ExpressionAttributeValues={':type': 'market_data'},
            Limit=1
        )
        
        items = response.get('Items', [])
        if items:
            return items[0].get('data', {})
        else:
            # Return default data if none found
            return get_default_market_data()
            
    except Exception as e:
        logger.error(f"Error getting recent market data: {str(e)}")
        return get_default_market_data()

def get_default_market_data() -> Dict[str, Any]:
    """Return default market data when no recent data is available"""
    return {
        'timestamp': int(datetime.utcnow().timestamp()),
        'sources': {
            'job_boards': collect_job_board_data(),
            'salary_data': collect_salary_data(),
            'skill_trends': collect_skill_trends()
        },
        'insights': {
            'market_health': 'Strong',
            'top_roles': ['Software Engineer', 'Data Scientist'],
            'salary_trends': 'Upward trend',
            'job_seeker_advice': ['Focus on in-demand skills']
        }
    }

def analyze_market_for_role(market_data: Dict, role: str, location: str, experience: str) -> Dict[str, Any]:
    """Analyze market data for a specific role and location"""
    try:
        sources = market_data.get('sources', {})
        job_boards = sources.get('job_boards', {})
        salary_data = sources.get('salary_data', {})
        skill_trends = sources.get('skill_trends', {})
        
        # Role-specific analysis
        role_jobs = job_boards.get('by_role', {}).get(role, 0)
        total_jobs = job_boards.get('total_jobs', 1)
        role_percentage = (role_jobs / total_jobs) * 100 if total_jobs > 0 else 0
        
        # Salary analysis
        role_salary = salary_data.get('by_role', {}).get(role, {})
        experience_salary = role_salary.get(experience, {})
        
        # Location adjustment
        location_data = salary_data.get('by_location', {}).get(location, {'multiplier': 1.0})
        adjusted_salary = {}
        if experience_salary:
            multiplier = location_data.get('multiplier', 1.0)
            adjusted_salary = {
                'min': int(experience_salary.get('min', 0) * multiplier),
                'max': int(experience_salary.get('max', 0) * multiplier),
                'median': int(experience_salary.get('median', 0) * multiplier)
            }
        
        # Relevant skills
        relevant_skills = []
        for skill_list in [skill_trends.get('trending_up', []), 
                          skill_trends.get('stable_high_demand', [])]:
            relevant_skills.extend([s['skill'] for s in skill_list[:5]])
        
        analysis = {
            'role': role,
            'location': location,
            'experience_level': experience,
            'market_overview': {
                'total_openings': role_jobs,
                'market_share': round(role_percentage, 1),
                'demand_level': 'High' if role_percentage > 10 else 'Medium' if role_percentage > 5 else 'Low',
                'growth_trend': 'Growing' if job_boards.get('growth_rate', 0) > 0.1 else 'Stable'
            },
            'salary_insights': {
                'range': adjusted_salary,
                'location_adjustment': location_data.get('multiplier', 1.0),
                'cost_of_living': location_data.get('cost_of_living', 1.0)
            },
            'skill_recommendations': relevant_skills[:8],
            'top_companies': job_boards.get('trending_companies', [])[:10],
            'recommendations': generate_role_recommendations(role, experience, market_data)
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing market for role: {str(e)}")
        return {
            'role': role,
            'location': location,
            'error': 'Analysis unavailable',
            'recommendations': ['Please try again later']
        }

def generate_role_recommendations(role: str, experience: str, market_data: Dict) -> List[str]:
    """Generate specific recommendations for a role and experience level"""
    
    recommendations = []
    
    # Experience-based recommendations
    if experience == 'entry':
        recommendations.extend([
            'Focus on building a strong portfolio with personal projects',
            'Contribute to open source projects to gain visibility',
            'Consider internships or entry-level positions at growing companies'
        ])
    elif experience == 'mid':
        recommendations.extend([
            'Develop leadership and mentoring skills',
            'Specialize in high-demand technologies',
            'Build a professional network in your target companies'
        ])
    else:  # senior
        recommendations.extend([
            'Focus on system design and architecture skills',
            'Consider roles with equity compensation',
            'Leverage your experience for consulting opportunities'
        ])
    
    # Role-specific recommendations
    if 'Engineer' in role:
        recommendations.append('Stay current with latest frameworks and tools')
    elif 'Data' in role:
        recommendations.append('Build expertise in AI/ML and cloud platforms')
    elif 'DevOps' in role:
        recommendations.append('Master container orchestration and infrastructure as code')
    
    return recommendations[:5]