import * as cdk from 'aws-cdk-lib';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as apigateway from 'aws-cdk-lib/aws-apigateway';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as cognito from 'aws-cdk-lib/aws-cognito';
import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';
import * as origins from 'aws-cdk-lib/aws-cloudfront-origins';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as stepfunctions from 'aws-cdk-lib/aws-stepfunctions';
import * as sfnTasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';
import * as sqs from 'aws-cdk-lib/aws-sqs';
import * as sns from 'aws-cdk-lib/aws-sns';
import { Construct } from 'constructs';

export class CareerPlatformStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // ========================================
    // STORAGE LAYER
    // ========================================

    // S3 Buckets
    const resumesBucket = new s3.Bucket(this, 'ResumesBucket', {
      bucketName: `career-platform-resumes-${this.account}`,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      lifecycleRules: [{
        id: 'DeleteOldResumes',
        expiration: cdk.Duration.days(365),
        transitions: [{
          storageClass: s3.StorageClass.INFREQUENT_ACCESS,
          transitionAfter: cdk.Duration.days(30)
        }]
      }]
    });

    const reportsBucket = new s3.Bucket(this, 'ReportsBucket', {
      bucketName: `career-platform-reports-${this.account}`,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL
    });

    const webAssetsBucket = new s3.Bucket(this, 'WebAssetsBucket', {
      bucketName: `career-platform-web-${this.account}`,
      encryption: s3.BucketEncryption.S3_MANAGED,
      websiteIndexDocument: 'index.html',
      websiteErrorDocument: 'error.html'
    });

    // DynamoDB Tables
    const usersTable = new dynamodb.Table(this, 'UsersTable', {
      tableName: 'career-platform-users',
      partitionKey: { name: 'userId', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      encryption: dynamodb.TableEncryption.AWS_MANAGED,
      pointInTimeRecovery: true,
      stream: dynamodb.StreamViewType.NEW_AND_OLD_IMAGES
    });

    const resumeAnalysisTable = new dynamodb.Table(this, 'ResumeAnalysisTable', {
      tableName: 'career-platform-resume-analysis',
      partitionKey: { name: 'analysisId', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'timestamp', type: dynamodb.AttributeType.NUMBER },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      encryption: dynamodb.TableEncryption.AWS_MANAGED
    });

    const jobMarketTable = new dynamodb.Table(this, 'JobMarketTable', {
      tableName: 'career-platform-job-market',
      partitionKey: { name: 'jobId', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'scrapedAt', type: dynamodb.AttributeType.NUMBER },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      encryption: dynamodb.TableEncryption.AWS_MANAGED,
      timeToLiveAttribute: 'ttl'
    });

    const learningPathsTable = new dynamodb.Table(this, 'LearningPathsTable', {
      tableName: 'career-platform-learning-paths',
      partitionKey: { name: 'userId', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'pathId', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      encryption: dynamodb.TableEncryption.AWS_MANAGED
    });

    // ========================================
    // AUTHENTICATION
    // ========================================

    const userPool = new cognito.UserPool(this, 'UserPool', {
      userPoolName: 'career-platform-users',
      selfSignUpEnabled: true,
      signInAliases: { email: true },
      passwordPolicy: {
        minLength: 8,
        requireLowercase: true,
        requireUppercase: true,
        requireDigits: true,
        requireSymbols: true
      },
      mfa: cognito.Mfa.OPTIONAL,
      mfaSecondFactor: {
        sms: true,
        otp: true
      },
      accountRecovery: cognito.AccountRecovery.EMAIL_ONLY,
      standardAttributes: {
        email: { required: true, mutable: true },
        givenName: { required: true, mutable: true },
        familyName: { required: true, mutable: true }
      }
    });

    const userPoolClient = new cognito.UserPoolClient(this, 'UserPoolClient', {
      userPool,
      authFlows: {
        userSrp: true,
        userPassword: true
      },
      generateSecret: false,
      refreshTokenValidity: cdk.Duration.days(30),
      accessTokenValidity: cdk.Duration.hours(1),
      idTokenValidity: cdk.Duration.hours(1)
    });

    // ========================================
    // MESSAGING & EVENTS
    // ========================================

    const resumeProcessingQueue = new sqs.Queue(this, 'ResumeProcessingQueue', {
      queueName: 'career-platform-resume-processing',
      visibilityTimeout: cdk.Duration.minutes(15),
      deadLetterQueue: {
        queue: new sqs.Queue(this, 'ResumeProcessingDLQ', {
          queueName: 'career-platform-resume-processing-dlq'
        }),
        maxReceiveCount: 3
      }
    });

    const notificationTopic = new sns.Topic(this, 'NotificationTopic', {
      topicName: 'career-platform-notifications'
    });

    // ========================================
    // LAMBDA FUNCTIONS
    // ========================================

    // Common Lambda layer for shared dependencies
    const commonLayer = new lambda.LayerVersion(this, 'CommonLayer', {
      code: lambda.Code.fromAsset('lambda-layers/common'),
      compatibleRuntimes: [lambda.Runtime.PYTHON_3_11],
      description: 'Common dependencies for Career Platform'
    });

    // Resume Processing Lambda
    const resumeProcessorFunction = new lambda.Function(this, 'ResumeProcessor', {
      runtime: lambda.Runtime.PYTHON_3_11,
      handler: 'resume_processor.handler',
      code: lambda.Code.fromAsset('lambda-functions/resume-processor'),
      layers: [commonLayer],
      timeout: cdk.Duration.minutes(5),
      memorySize: 1024,
      environment: {
        RESUMES_BUCKET: resumesBucket.bucketName,
        ANALYSIS_TABLE: resumeAnalysisTable.tableName,
        NOTIFICATION_TOPIC: notificationTopic.topicArn
      }
    });

    // Job Market Intelligence Lambda
    const jobMarketFunction = new lambda.Function(this, 'JobMarketIntelligence', {
      runtime: lambda.Runtime.PYTHON_3_11,
      handler: 'job_market.handler',
      code: lambda.Code.fromAsset('lambda-functions/job-market'),
      layers: [commonLayer],
      timeout: cdk.Duration.minutes(10),
      memorySize: 2048,
      environment: {
        JOB_MARKET_TABLE: jobMarketTable.tableName
      }
    });

    // AI Career Coach Lambda
    const careerCoachFunction = new lambda.Function(this, 'CareerCoach', {
      runtime: lambda.Runtime.PYTHON_3_11,
      handler: 'career_coach.handler',
      code: lambda.Code.fromAsset('lambda-functions/career-coach'),
      layers: [commonLayer],
      timeout: cdk.Duration.minutes(2),
      memorySize: 1024,
      environment: {
        USERS_TABLE: usersTable.tableName,
        LEARNING_PATHS_TABLE: learningPathsTable.tableName
      }
    });

    // Learning Path Generator Lambda
    const learningPathFunction = new lambda.Function(this, 'LearningPathGenerator', {
      runtime: lambda.Runtime.PYTHON_3_11,
      handler: 'learning_path.handler',
      code: lambda.Code.fromAsset('lambda-functions/learning-path'),
      layers: [commonLayer],
      timeout: cdk.Duration.minutes(3),
      memorySize: 1024,
      environment: {
        LEARNING_PATHS_TABLE: learningPathsTable.tableName,
        ANALYSIS_TABLE: resumeAnalysisTable.tableName
      }
    });

    // User Management Lambda
    const userManagementFunction = new lambda.Function(this, 'UserManagement', {
      runtime: lambda.Runtime.PYTHON_3_11,
      handler: 'user_management.handler',
      code: lambda.Code.fromAsset('lambda-functions/user-management'),
      layers: [commonLayer],
      timeout: cdk.Duration.seconds(30),
      memorySize: 512,
      environment: {
        USERS_TABLE: usersTable.tableName,
        USER_POOL_ID: userPool.userPoolId
      }
    });

    // ========================================
    // STEP FUNCTIONS WORKFLOW
    // ========================================

    const resumeAnalysisWorkflow = new stepfunctions.StateMachine(this, 'ResumeAnalysisWorkflow', {
      stateMachineName: 'career-platform-resume-analysis',
      definition: new sfnTasks.LambdaInvoke(this, 'ProcessResume', {
        lambdaFunction: resumeProcessorFunction,
        outputPath: '$.Payload'
      }).next(
        new stepfunctions.Parallel(this, 'ParallelAnalysis')
          .branch(
            new sfnTasks.LambdaInvoke(this, 'GenerateLearningPath', {
              lambdaFunction: learningPathFunction
            })
          )
          .branch(
            new sfnTasks.LambdaInvoke(this, 'GetJobMarketInsights', {
              lambdaFunction: jobMarketFunction
            })
          )
      ).next(
        new sfnTasks.SnsPublish(this, 'NotifyCompletion', {
          topic: notificationTopic,
          message: stepfunctions.TaskInput.fromText('Resume analysis completed')
        })
      )
    });

    // ========================================
    // API GATEWAY
    // ========================================

    const api = new apigateway.RestApi(this, 'CareerPlatformAPI', {
      restApiName: 'Career Platform API',
      description: 'API for AI-Powered Career Development Platform',
      defaultCorsPreflightOptions: {
        allowOrigins: apigateway.Cors.ALL_ORIGINS,
        allowMethods: apigateway.Cors.ALL_METHODS,
        allowHeaders: ['Content-Type', 'Authorization']
      },
      deployOptions: {
        stageName: 'prod',
        throttlingRateLimit: 1000,
        throttlingBurstLimit: 2000,
        loggingLevel: apigateway.MethodLoggingLevel.INFO,
        dataTraceEnabled: true,
        metricsEnabled: true
      }
    });

    // Cognito Authorizer
    const authorizer = new apigateway.CognitoUserPoolsAuthorizer(this, 'APIAuthorizer', {
      cognitoUserPools: [userPool],
      authorizerName: 'CareerPlatformAuthorizer'
    });

    // API Resources
    const usersResource = api.root.addResource('users');
    const resumesResource = api.root.addResource('resumes');
    const coachResource = api.root.addResource('coach');
    const learningResource = api.root.addResource('learning');
    const jobsResource = api.root.addResource('jobs');

    // User Management Endpoints
    usersResource.addMethod('GET', new apigateway.LambdaIntegration(userManagementFunction), {
      authorizer,
      authorizationType: apigateway.AuthorizationType.COGNITO
    });

    usersResource.addMethod('POST', new apigateway.LambdaIntegration(userManagementFunction), {
      authorizer,
      authorizationType: apigateway.AuthorizationType.COGNITO
    });

    // Resume Analysis Endpoints
    resumesResource.addMethod('POST', new apigateway.LambdaIntegration(resumeProcessorFunction), {
      authorizer,
      authorizationType: apigateway.AuthorizationType.COGNITO
    });

    // Career Coach Endpoints
    coachResource.addMethod('POST', new apigateway.LambdaIntegration(careerCoachFunction), {
      authorizer,
      authorizationType: apigateway.AuthorizationType.COGNITO
    });

    // Learning Path Endpoints
    learningResource.addMethod('GET', new apigateway.LambdaIntegration(learningPathFunction), {
      authorizer,
      authorizationType: apigateway.AuthorizationType.COGNITO
    });

    // Job Market Endpoints
    jobsResource.addMethod('GET', new apigateway.LambdaIntegration(jobMarketFunction));

    // ========================================
    // CLOUDFRONT DISTRIBUTION
    // ========================================

    const distribution = new cloudfront.Distribution(this, 'WebDistribution', {
      defaultBehavior: {
        origin: new origins.S3Origin(webAssetsBucket),
        viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        cachePolicy: cloudfront.CachePolicy.CACHING_OPTIMIZED,
        compress: true
      },
      additionalBehaviors: {
        '/api/*': {
          origin: new origins.RestApiOrigin(api),
          viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.HTTPS_ONLY,
          cachePolicy: cloudfront.CachePolicy.CACHING_DISABLED,
          originRequestPolicy: cloudfront.OriginRequestPolicy.CORS_S3_ORIGIN
        }
      },
      defaultRootObject: 'index.html',
      errorResponses: [{
        httpStatus: 404,
        responseHttpStatus: 200,
        responsePagePath: '/index.html'
      }]
    });

    // ========================================
    // EVENT RULES
    // ========================================

    // Daily job market data collection
    new events.Rule(this, 'DailyJobMarketUpdate', {
      schedule: events.Schedule.cron({ hour: '2', minute: '0' }),
      targets: [new targets.LambdaFunction(jobMarketFunction)]
    });

    // ========================================
    // IAM PERMISSIONS
    // ========================================

    // Resume Processor Permissions
    resumesBucket.grantReadWrite(resumeProcessorFunction);
    resumeAnalysisTable.grantWriteData(resumeProcessorFunction);
    notificationTopic.grantPublish(resumeProcessorFunction);

    // Job Market Function Permissions
    jobMarketTable.grantWriteData(jobMarketFunction);

    // Career Coach Permissions
    usersTable.grantReadData(careerCoachFunction);
    learningPathsTable.grantReadWriteData(careerCoachFunction);

    // Learning Path Generator Permissions
    learningPathsTable.grantWriteData(learningPathFunction);
    resumeAnalysisTable.grantReadData(learningPathFunction);

    // User Management Permissions
    usersTable.grantReadWriteData(userManagementFunction);

    // Bedrock permissions for AI functions
    const bedrockPolicy = new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'bedrock:InvokeModel',
        'bedrock:InvokeModelWithResponseStream'
      ],
      resources: ['*']
    });

    careerCoachFunction.addToRolePolicy(bedrockPolicy);
    learningPathFunction.addToRolePolicy(bedrockPolicy);

    // Comprehend permissions
    const comprehendPolicy = new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'comprehend:DetectEntities',
        'comprehend:DetectKeyPhrases',
        'comprehend:DetectSentiment'
      ],
      resources: ['*']
    });

    resumeProcessorFunction.addToRolePolicy(comprehendPolicy);

    // Textract permissions
    const textractPolicy = new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'textract:DetectDocumentText',
        'textract:AnalyzeDocument'
      ],
      resources: ['*']
    });

    resumeProcessorFunction.addToRolePolicy(textractPolicy);

    // Step Functions permissions
    resumeAnalysisWorkflow.grantStartExecution(resumeProcessorFunction);

    // ========================================
    // OUTPUTS
    // ========================================

    new cdk.CfnOutput(this, 'APIEndpoint', {
      value: api.url,
      description: 'API Gateway endpoint URL'
    });

    new cdk.CfnOutput(this, 'CloudFrontURL', {
      value: `https://${distribution.distributionDomainName}`,
      description: 'CloudFront distribution URL'
    });

    new cdk.CfnOutput(this, 'UserPoolId', {
      value: userPool.userPoolId,
      description: 'Cognito User Pool ID'
    });

    new cdk.CfnOutput(this, 'UserPoolClientId', {
      value: userPoolClient.userPoolClientId,
      description: 'Cognito User Pool Client ID'
    });

    new cdk.CfnOutput(this, 'ResumesBucketName', {
      value: resumesBucket.bucketName,
      description: 'S3 bucket for resume storage'
    });
  }
}

const app = new cdk.App();
new CareerPlatformStack(app, 'CareerPlatformStack', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION
  }
});