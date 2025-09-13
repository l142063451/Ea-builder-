--- FILENAME: Copilot_Instructions.md ---

# Copilot Agent Instructions for Forex Autonomous Trading Bot

## Primary Directive

You are an advanced autonomous project continuation agent. Your mission is to build a world-class Forex trading bot that achieves ≥90% accuracy and consistent profitability. These instructions serve as your permanent memory and operational guidelines.

## Core Operating Principles

### 1. Memory-First Operation
- **ALWAYS** read `Requirements_and_Goals.md` before starting any new work
- **ALWAYS** check `Status.md` to understand current stage and next tasks
- **NEVER** proceed without understanding the project's current state
- **NEVER** skip reading the memory files - they contain essential context

### 2. Documentation and Logging
- **ALWAYS** append results to `Status.md` after completing tasks
- **ALWAYS** update performance metrics with actual measurements
- **ALWAYS** log decisions, results, and reasoning in structured format
- **NEVER** work silently - document everything you do

### 3. Quality and Validation
- **ALWAYS** run tests and backtests before considering a task complete
- **ALWAYS** validate against defined success criteria
- **NEVER** advance to next stage without meeting performance gates
- **NEVER** use placeholders or mock implementations

## Operational Workflow

### Starting a New Session
1. Read `Requirements_and_Goals.md` to understand the complete vision
2. Read `Status.md` to identify current stage and pending tasks
3. Review recent validation results and performance metrics
4. Identify the highest priority pending task
5. Check for any blockers or dependencies
6. Proceed with implementation following these instructions

### Task Execution Protocol
1. **Plan**: Create detailed implementation plan with success criteria
2. **Implement**: Write production-ready code with proper error handling
3. **Test**: Create and run comprehensive tests
4. **Validate**: Measure performance against defined criteria
5. **Document**: Update Status.md with results and metrics
6. **Review**: Assess if stage completion criteria are met

## Implementation Standards

### Code Quality Requirements
- **Production Ready**: All code must be production-grade, not prototype
- **Error Handling**: Comprehensive exception handling and recovery
- **Performance**: Meet latency and throughput requirements
- **Testing**: >80% test coverage for all components
- **Documentation**: Inline comments and docstrings for complex logic
- **Type Hints**: Python type annotations for all functions

### Data and Model Standards
- **Data Validation**: Always validate data quality before use
- **Model Validation**: Cross-validation, walk-forward testing
- **Performance Metrics**: Track and log all defined KPIs
- **Overfitting Prevention**: Proper train/validation/test splits
- **Reproducibility**: Set random seeds, version dependencies

### Trading System Standards
- **Risk First**: Implement risk management before profit optimization
- **Simulation**: Thorough backtesting before live deployment
- **Monitoring**: Real-time performance and risk monitoring
- **Circuit Breakers**: Automated stop mechanisms for risk control
- **Audit Trail**: Complete logging of all trading decisions

## Stage-Specific Instructions

### Data Collection and Preparation Stage
**Entry Criteria**: Project initialized with memory files
**Success Criteria**: >99% data completeness, database operational

**Required Tasks**:
1. Setup PostgreSQL database with optimized schema
2. Download historical data for EURUSD, GBPUSD, USDJPY, AUDUSD (5 years, 1-minute)
3. Implement data quality validation and cleaning
4. Create real-time data feed connection
5. Build data access APIs with caching

**Validation Steps**:
- Verify data completeness >99%
- Test database query performance <50ms for 100K records
- Confirm real-time feed latency <1 second
- Generate data quality report

**Advancement Criteria**:
- All 4 currency pairs data collected
- Database queries optimized
- Real-time feed stable
- Data quality score >99%

### Feature Engineering Stage
**Entry Criteria**: Data stage completed successfully
**Success Criteria**: Feature importance analysis showing >0.1 correlation to returns

**Required Tasks**:
1. Implement basic technical indicators (SMA, EMA, RSI, MACD, BB, ATR)
2. Add advanced indicators (Stochastic, Williams %R, CCI)
3. Create multi-timeframe feature matrix (M1, M5, M15, H1, H4, D1)
4. Implement price pattern recognition
5. Perform feature importance analysis

**Validation Steps**:
- Compare indicator calculations vs. TradingView (100% accuracy)
- Generate feature correlation matrix
- Perform statistical significance tests
- Create feature importance rankings

**Advancement Criteria**:
- All technical indicators implemented and validated
- Multi-timeframe coherence established
- Feature importance analysis completed
- Top 20 features identified with statistical significance

### Model Development Stage
**Entry Criteria**: Features engineered and validated
**Success Criteria**: Model accuracy >85% on out-of-sample data

**Required Tasks**:
1. Implement baseline models (logistic regression, random forest)
2. Develop neural networks (LSTM, GRU, Transformer)
3. Create reinforcement learning agents (PPO, SAC)
4. Build ensemble methods with dynamic weighting
5. Perform hyperparameter optimization

**Validation Steps**:
- Cross-validation with walk-forward analysis
- Out-of-sample testing on recent data
- Overfitting detection and mitigation
- Model interpretability analysis

**Advancement Criteria**:
- Baseline accuracy >60%
- Advanced model accuracy >85%
- Ensemble accuracy >90%
- Overfitting checks passed

### Backtesting Stage
**Entry Criteria**: Models trained and validated
**Success Criteria**: Sharpe ratio >2.0, max drawdown <10%

**Required Tasks**:
1. Implement realistic execution simulation (spreads, slippage, commission)
2. Create walk-forward backtesting framework
3. Perform Monte Carlo analysis
4. Test multiple market regimes
5. Generate comprehensive performance report

**Validation Steps**:
- Minimum 1000 trades in backtest
- Test on different time periods
- Stress test with extreme market conditions
- Validate risk management effectiveness

**Advancement Criteria**:
- Sharpe ratio >2.0
- Maximum drawdown <10%
- Win rate >65%
- Profit factor >1.5

### Paper Trading Stage
**Entry Criteria**: Backtesting requirements met
**Success Criteria**: Live performance within 5% of backtest results

**Required Tasks**:
1. Implement paper trading engine
2. Connect to live market data feeds
3. Monitor execution latency and slippage
4. Compare live vs. backtest performance
5. Optimize execution algorithms

**Validation Steps**:
- Run paper trading for minimum 30 days
- Monitor execution quality metrics
- Track performance vs. backtest expectations
- Analyze slippage and latency patterns

**Advancement Criteria**:
- System uptime >99%
- Execution latency <50ms
- Performance correlation >95% with backtest
- Risk limits consistently respected

### Live Deployment Stage
**Entry Criteria**: Paper trading validation successful
**Success Criteria**: Human approval and regulatory compliance

**Required Tasks**:
1. Implement human approval workflow
2. Create risk monitoring dashboard
3. Setup automated alerting system
4. Prepare regulatory compliance documentation
5. Create deployment procedures

**Validation Steps**:
- Security audit of system components
- Risk management testing
- Disaster recovery testing
- Compliance verification

**Advancement Criteria**:
- Human approval obtained
- Security audit passed
- Risk controls validated
- Regulatory compliance confirmed

## Advanced Strategy Implementation

### Multi-Strategy Portfolio Management
- **Strategy Allocation**: Dynamic allocation based on market conditions
- **Correlation Management**: Maintain <70% correlation between strategies
- **Performance Tracking**: Individual strategy performance monitoring
- **Rebalancing**: Daily portfolio rebalancing based on recent performance

### Memory-Based Decision Making
- **Pattern Matching**: Use FAISS for similar market condition identification
- **Historical Analysis**: Query past performance in similar conditions
- **Adaptive Learning**: Update strategy weights based on recent performance
- **Risk Memory**: Remember and avoid historical high-risk scenarios

### Advanced Risk Management
- **Kelly Criterion**: Dynamic position sizing based on edge and odds
- **Volatility Scaling**: Position size inverse to market volatility
- **Drawdown Control**: Progressive position reduction during drawdowns
- **Circuit Breakers**: Multiple levels of automated trading halts

## Performance Optimization Guidelines

### Execution Optimization
- **Order Routing**: Smart order routing to minimize slippage
- **Timing**: Market microstructure analysis for optimal entry timing
- **Size Optimization**: Position sizing for minimal market impact
- **Latency**: Sub-50ms execution from signal to order

### Model Performance
- **Ensemble Methods**: Combine multiple models for robustness
- **Online Learning**: Continuous model updates with new data
- **Regime Detection**: Adapt strategies to market regime changes
- **Feature Selection**: Dynamic feature selection based on relevance

### System Performance
- **Database Optimization**: Indexed queries, connection pooling
- **Memory Management**: Efficient data structures and caching
- **Parallel Processing**: Multi-threading for data processing
- **Monitoring**: Real-time system health monitoring

## Quality Assurance Protocols

### Testing Requirements
- **Unit Tests**: Test all individual components
- **Integration Tests**: Test component interactions
- **Performance Tests**: Validate latency and throughput
- **Stress Tests**: Test under extreme conditions
- **Regression Tests**: Ensure changes don't break existing functionality

### Validation Procedures
- **Data Validation**: Check data integrity and completeness
- **Model Validation**: Cross-validation and out-of-sample testing
- **Strategy Validation**: Backtest with multiple scenarios
- **System Validation**: End-to-end testing of complete system

### Documentation Standards
- **Code Documentation**: Comprehensive docstrings and comments
- **API Documentation**: Clear interface documentation
- **User Documentation**: Operational procedures and guides
- **Technical Documentation**: Architecture and design decisions

## Continuous Learning Framework

### Model Updates
- **Daily Retraining**: Update models with latest market data
- **Performance Monitoring**: Track model degradation over time
- **A/B Testing**: Compare model versions in paper trading
- **Rollback Procedures**: Quick rollback for underperforming models

### Strategy Evolution
- **Performance Analysis**: Regular strategy performance reviews
- **Parameter Optimization**: Continuous hyperparameter tuning
- **New Strategy Integration**: Framework for adding new strategies
- **Strategy Retirement**: Remove consistently underperforming strategies

### System Improvements
- **Performance Profiling**: Regular system performance analysis
- **Bottleneck Identification**: Identify and resolve performance bottlenecks
- **Scalability Planning**: Plan for increased trading volume
- **Technology Updates**: Regular updates of core dependencies

## Error Handling and Recovery

### System Failures
- **Connection Failures**: Automatic reconnection with exponential backoff
- **Data Feed Failures**: Failover to backup data sources
- **Order Failures**: Retry logic with position verification
- **System Crashes**: Automated restart with state recovery

### Performance Degradation
- **Accuracy Decline**: Automatic model retraining triggers
- **Execution Issues**: Automatic adjustment of execution parameters
- **Risk Limit Breaches**: Immediate position reduction protocols
- **Market Regime Changes**: Strategy adaptation procedures

### Human Intervention Triggers
- **System Failures**: Critical system component failures
- **Performance Alerts**: Significant performance degradation
- **Risk Breaches**: Major risk limit violations
- **Regulatory Issues**: Compliance-related concerns

## Security and Compliance

### Security Measures
- **API Key Management**: Encrypted storage of sensitive credentials
- **Access Control**: Role-based access to system components
- **Audit Logging**: Comprehensive logging of all system activities
- **Network Security**: VPN connections to broker APIs

### Compliance Requirements
- **Trade Reporting**: Regulatory trade reporting capabilities
- **Record Keeping**: Maintain all required trading records
- **Risk Reporting**: Regular risk position reporting
- **Audit Trail**: Complete audit trail for all trading activities

## Emergency Procedures

### Trading Halt Procedures
1. **Immediate**: Stop all new position entries
2. **Assess**: Evaluate current positions and market conditions
3. **Decide**: Determine whether to close positions or maintain
4. **Execute**: Implement decision with appropriate risk controls
5. **Document**: Log all actions and decisions

### System Recovery Procedures
1. **Identify**: Quickly identify the source of the problem
2. **Isolate**: Isolate affected components to prevent spread
3. **Recover**: Implement recovery procedures for affected components
4. **Verify**: Verify system integrity after recovery
5. **Resume**: Resume normal operations with enhanced monitoring

## Success Metrics and KPIs

### Primary Metrics (Daily Monitoring)
- **Accuracy**: >90% signal accuracy
- **Returns**: Daily P&L tracking
- **Drawdown**: Current drawdown level
- **Risk Metrics**: VaR, expected shortfall
- **Execution**: Average execution latency

### Secondary Metrics (Weekly Monitoring)
- **Sharpe Ratio**: Risk-adjusted returns
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss
- **Maximum Drawdown**: Peak-to-trough decline
- **System Uptime**: System availability percentage

### Strategic Metrics (Monthly Monitoring)
- **Model Accuracy**: Predictive accuracy trends
- **Strategy Performance**: Individual strategy returns
- **Risk-Adjusted Returns**: Various risk-adjusted metrics
- **Market Share**: Trading volume relative to market
- **Continuous Improvement**: Performance enhancement rate

## Final Reminders

### Critical Rules
- **NEVER** trade with real money without human approval
- **NEVER** skip validation and testing phases
- **NEVER** implement placeholder or mock code
- **NEVER** proceed without meeting stage completion criteria
- **ALWAYS** prioritize risk management over profit optimization
- **ALWAYS** maintain complete audit trails
- **ALWAYS** test thoroughly before deployment

### Success Vision
Build a fully autonomous, profitable, and safe Forex trading system that:
- Operates with minimal human intervention
- Achieves consistent profitability with controlled risk
- Learns and adapts to changing market conditions
- Maintains the highest standards of reliability and auditability
- Serves as a benchmark for algorithmic trading excellence

---

**Remember**: You are building a production system that will trade real money. Every decision must prioritize capital preservation while seeking consistent profitability. The memory files are your persistent knowledge base - use them effectively to maintain continuity and achieve the project's ambitious goals.