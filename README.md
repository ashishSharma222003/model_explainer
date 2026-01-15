# Model Explainer - AI-Powered Fraud Analytics Platform

An intelligent analytics platform that uses AI to discover hidden decision patterns in fraud detection systems, generate human-readable shadow rules, and produce actionable compliance reports.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-green.svg)
![Next.js](https://img.shields.io/badge/next.js-14-black.svg)
![TypeScript](https://img.shields.io/badge/typescript-5.0-blue.svg)

## 🎯 What It Does

Model Explainer analyzes fraud analyst decisions to:
- **Discover Shadow Rules**: AI automatically identifies undocumented decision patterns that analysts follow
- **Generate Business Reports**: Creates executive-friendly reports with compliance analysis
- **Visualize Decision Trees**: Shows how analysts make fraud/legit decisions using Random Forest analysis
- **Check Guideline Compliance**: Compares discovered patterns against official bank policies
- **Predict Fraud**: Analyzes transaction data to identify fraud patterns and accuracy metrics

## ✨ Key Features

### 1. **Discovery Section** 
- AI-generated shadow rules in plain business language
- Expandable table showing:
  - Rule descriptions for non-technical stakeholders
  - Coverage % and transaction counts
  - Accuracy metrics with confidence levels
  - Technical rule details (for reference)
- One-click rule regeneration

### 2. **Insights Dashboard**
- Random Forest analysis of L1 and L2 analyst decisions
- Feature importance visualization
- Prediction accuracy breakdown (true/false positives/negatives)
- Decision tree visualization
- Wrong predictions analysis
- Hyperparameter tuning

### 3. **Reports Section**
Two AI-generated report types:

**Business Impact Report** (for executives)
- Executive summary with key findings
- Guideline compliance analysis
- Areas of concern and improvement opportunities
- Strategic recommendations with ROI estimates
- No technical jargon - pure business language

**Analyst Performance Report** (for compliance teams)
- Shadow rule detection with severity levels
- Bias analysis (demographic, temporal, amount-based)
- Guideline compliance scoring
- Training recommendations
- Process improvement suggestions

### 4. **Guideline Compliance**
- Configure bank policies and guidelines
- Automatic compliance checking in reports
- Violation flagging with risk levels (Low/Medium/High/Critical)
- Specific recommendations for each violation

## 🏗️ Tech Stack

### Backend
- **FastAPI** - High-performance Python web framework
- **LangChain** + **Anthropic Claude** - LLM-powered analysis
- **Scikit-learn** - Random Forest modeling
- **Pandas** - Data processing
- **Pydantic** - Data validation

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Modern styling
- **Framer Motion** - Smooth animations
- **Lucide React** - Beautiful icons

### Infrastructure
- **Docker** + **Docker Compose** - Containerization
- **FAISS** - Vector similarity search for shadow rules

## 🚀 Quick Start

### Prerequisites
- Docker Desktop installed
- Anthropic API key ([get one here](https://console.anthropic.com/))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ashishSharma222003/model_explainer.git
cd model_explainer
```

2. **Set up environment variables**
```bash
# Create .env file in backend directory
cd backend
echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
cd ..
```

3. **Start the application**
```bash
docker-compose up --build
```

4. **Access the application**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 📖 Usage Guide

### Step 1: Upload Your Data
1. Navigate to the **Data** section
2. Upload a CSV file with columns:
   - `alert_id` - Unique transaction identifier
   - `l1_decision` - L1 analyst decision (0=Legit, 1=Fraud)
   - `l2_decision` - L2 analyst decision (0=Legit, 1=Fraud)
   - `true_fraud_flag` - Actual fraud status
   - Additional feature columns (transaction amount, merchant, etc.)

### Step 2: Configure Data Schema
Define your data columns:
- **Categorical**: Region, merchant type, customer segment
- **Numerical**: Transaction amount, account age, velocity
- **Datetime**: Transaction timestamp
- **Target**: `true_fraud_flag`

### Step 3: Add Guidelines (Optional)
Configure bank policies in the **Guidelines** section:
- High-value transaction protocols
- AML compliance rules
- Risk assessment criteria

### Step 4: Run Analysis
Navigate to **Insights** and click "Run Analysis":
- Generates Random Forest model
- Extracts decision patterns
- Calculates accuracy metrics
- Identifies wrong predictions

### Step 5: Discover Shadow Rules
Go to **Discovery** section:
- View AI-generated business rules
- See coverage and accuracy for each rule
- Click rows to expand technical details
- Regenerate rules anytime

### Step 6: Generate Reports
Visit **Reports** section and choose:
- **Business Impact Report** - For CEO/Board
- **Analyst Performance Report** - For Compliance/Training

Reports automatically include:
- Discovered shadow rules
- Guideline compliance analysis
- Specific violation flagging
- Actionable recommendations

## 📁 Project Structure

```
model_explainer/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── chat.py                 # LLM chat and rule generation
│   ├── xgboost_analyzer.py     # Random Forest analysis
│   ├── report_generator.py     # Report generation with LLM
│   ├── models.py               # Pydantic data models
│   └── session_manager.py      # Session management
├── frontend/
│   ├── app/
│   │   └── page.tsx            # Main application page
│   ├── components/
│   │   ├── DataSchemaInput.tsx      # Data configuration
│   │   ├── ResultsDashboard.tsx     # Insights dashboard
│   │   ├── DiscoverySection.tsx     # Shadow rules table
│   │   ├── ExecutiveReportSection.tsx # Reports UI
│   │   └── GuidelinesInput.tsx      # Guidelines configuration
│   └── lib/
│       └── api.ts              # API client
└── docker-compose.yml          # Docker orchestration
```

## 🎨 Features in Detail

### AI-Powered Shadow Rule Generation
Uses Claude (Anthropic) to translate technical decision tree patterns into business-friendly rules:
- **Context-Aware**: Includes Random Forest feature importance, accuracy metrics, and data schema
- **Deduplication**: Automatically consolidates similar patterns
- **Quality Filtering**: Only shows high-confidence, high-coverage rules
- **Business Language**: "High-value transactions from new merchants" instead of "Amount > 500 & merchant_new == 1"

### Guideline Compliance Checking
Reports automatically compare discovered patterns against official policies:
- Marks rules as: ✅ Aligned / ❌ Violates / ⚠️ Not Covered
- Specifies which guideline is violated
- Estimates compliance risk (Low/Medium/High/Critical)
- Recommends specific actions (training, updates, enforcement)

### Session Management
- Multiple sessions with auto-save
- Session switching without data loss
- Past report history
- CSV data persistence

## 🔧 Configuration

### Environment Variables

**Backend (.env)**
```bash
ANTHROPIC_API_KEY=sk-ant-...
LOG_LEVEL=INFO
```

### Hyperparameters (in UI)
- Number of trees in Random Forest
- Max depth
- Min samples split/leaf
- Feature subsampling

## 📊 Data Format

Your CSV should include:
```csv
alert_id,l1_decision,l2_decision,true_fraud_flag,amount,merchant_type,customer_age_days
TXN001,1,1,1,5000,new,15
TXN002,0,0,0,50,established,360
...
```

**Required Columns:**
- `alert_id`, `l1_decision`, `l2_decision`, `true_fraud_flag`

**Recommended Columns:**
- Transaction features: amount, merchant info, customer data, timestamps

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with [Anthropic Claude](https://www.anthropic.com/) for LLM capabilities
- UI inspired by modern fintech dashboards
- Random Forest implementation using [scikit-learn](https://scikit-learn.org/)

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Contact: ashishSharma222003@github.com

---

**Made with ❤️ for better fraud detection transparency**
