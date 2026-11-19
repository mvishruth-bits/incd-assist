"""
Chakritha's five classical agents for incident analysis.
"""
import json
from pathlib import Path
from incd_assist.contracts import Agent, AgentResult, IncidentScenario
PROJECT_ROOT = Path(__file__).resolve().parent.parent

class SimpleReflexAgent:

    def decide(self, scenario: IncidentScenario) -> AgentResult:
        metrics = scenario.metrics
        logs = scenario.logs
        latency = metrics.get('latency_p95_ms', 0)
        error_rate = metrics.get('error_rate_pct', 0)
        cpu = metrics.get('cpu_pct', 0)
        requests = metrics.get('requests_per_min', 0)
        db_connections = metrics.get('db_connections_in_use', 0)
        db_limit = metrics.get('db_pool_limit', 0)
        availability = metrics.get('availability_pct', 0)
        log_text = ' '.join(logs).lower()
        if db_limit > 0 and db_connections >= db_limit and (latency >= 1000 or error_rate >= 5):
            cause = 'Database connection pool exhaustion'
            summary = 'API requests are experiencing delays because the database connection pool is fully occupied.'
            action = 'Investigate long-running database connections, connection wait queues and pool capacity. Review safe capacity adjustments.'
            evidence = (f'Database connections: {db_connections}/{db_limit}', f'P95 latency: {latency} ms', f'Error rate: {error_rate}%', 'Database pool exhaustion detected by predefined rule.')
            reason = 'Rule 1 matched: Database pool is full.'
        elif 'startup failed' in log_text or 'missing auth_issuer' in log_text:
            deployments = [dict(item) for item in scenario.deployment_history]
            previous = dict(scenario.previous_version)
            failed = next((item for item in reversed(deployments) if 'failed' in str(item.get('status', '')).lower()), {})
            failed_version = failed.get('version', 'unknown')
            failed_change = failed.get('change', 'unknown')
            previous_version = previous.get('version', 'unknown')
            cause = 'Application startup failure due to AUTH_ISSUER configuration change'
            summary = f'Application deployment {failed_version} failed to start because the AUTH_ISSUER configuration is missing.'
            action = f'Compare the failed configuration with the previous healthy version {previous_version}. After operator approval, restore the correct configuration or perform a controlled rollback. Verify application startup and health metrics.'
            evidence = (f'Failed deployment: {failed_version}', f'Configuration change: {failed_change}', f'Previous healthy version: {previous_version}', f"Previous version health: {previous.get('health')}", f"Previous configuration: {previous.get('config')}", 'Startup failure found in application logs. The earlier DB transient timeout was marked resolved; it is historical and inconclusive for the current startup failure.')
            reason = 'Rule 2 matched: Missing AUTH_ISSUER configuration after deployment.'
        elif cpu >= 85 and requests >= 4000 and (db_limit > 0) and (db_connections < db_limit * 0.8):
            cause = 'High traffic causing CPU saturation'
            summary = 'Incoming traffic is increasing CPU utilization while database connections remain healthy.'
            action = 'Review CPU utilization, request handling and horizontal scaling capacity.'
            evidence = (f'CPU utilization: {cpu}%', f'Requests per minute: {requests}', f'Database connections: {db_connections}/{db_limit}', 'Database pool is not exhausted.')
            reason = 'Rule 3 matched: High traffic and CPU utilization.'
        else:
            cause = 'Unknown or insufficiently classified incident'
            summary = 'Available evidence does not match known rules.'
            action = 'Collect additional monitoring metrics and log evidence before recommending corrective action.'
            evidence = (f'P95 latency: {latency} ms', f'CPU utilization: {cpu}%', f'Error rate: {error_rate}%')
            reason = 'No predefined rule matched.'
        return AgentResult(agent_name='Simple Reflex Agent', incident_summary=summary, evidence=evidence, likely_cause=cause, recommended_action=action, verification_steps=('Review the relevant monitoring metrics.', 'Inspect the corresponding application logs.', 'Confirm recovery using availability and error-rate metrics.'), limitations=('Uses predefined IF-THEN rules only.', 'Does not learn from previous incidents.', 'Does not maintain an internal memory or historical state.'), reason=reason)

class ModelBasedReflexAgent:

    def __init__(self):
        self.internal_state = {}

    def decide(self, scenario: IncidentScenario) -> AgentResult:
        sid = scenario.id
        metrics = scenario.metrics
        logs = list(scenario.logs)
        log_text = ' '.join(logs).lower()
        previous = self.internal_state.get(sid, {})
        observation_count = previous.get('observation_count', 0) + 1
        latency = metrics.get('latency_p95_ms', 0)
        error_rate = metrics.get('error_rate_pct', 0)
        cpu = metrics.get('cpu_pct', 0)
        requests = metrics.get('requests_per_min', 0)
        db_connections = metrics.get('db_connections_in_use', 0)
        db_limit = metrics.get('db_pool_limit', 0)
        evidence = []
        if 'startup failed' in log_text or 'missing auth_issuer' in log_text:
            deployments = [dict(item) for item in scenario.deployment_history]
            previous_version = dict(scenario.previous_version)
            failed = next((item for item in reversed(deployments) if 'failed' in item.get('status', '').lower()), {})
            failed_version = failed.get('version', 'unknown')
            healthy_version = previous_version.get('version', 'unknown')
            cause = 'Startup failure caused by missing AUTH_ISSUER configuration'
            summary = f'Deployment {failed_version} failed during startup following a configuration change.'
            action = f'Verify AUTH_ISSUER and compare the configuration with the previous healthy version {healthy_version}. After operator approval, restore the last-known-good configuration if the failure is confirmed.'
            evidence.extend([f'Failed deployment: {failed_version}', f'Previous healthy version: {healthy_version}', f"Previous configuration: {previous_version.get('config')}"])
            stale_db_warning = any(('db transient timeout' in log.lower() and 'resolved' in log.lower() for log in logs))
            if stale_db_warning:
                evidence.append('The earlier DB transient timeout was marked resolved; it is historical and inconclusive for the current startup failure.')
            reason = 'Internal model identified a configuration failure after comparing the deployment history and log timeline.'
        elif db_limit > 0 and db_connections >= db_limit and (latency >= 1000 or error_rate >= 5):
            cause = 'Database connection pool exhaustion'
            summary = 'The database connection pool is fully occupied, causing API delays and errors.'
            action = 'Investigate long-running connections and checkout queues. Review safe pool capacity adjustments.'
            evidence.extend([f'Database connections: {db_connections}/{db_limit}', f'P95 latency: {latency} ms', f'Error rate: {error_rate}%'])
            reason = 'Internal model identified database pool saturation.'
        elif cpu >= 85 and requests >= 4000 and (db_limit > 0) and (db_connections < db_limit * 0.8):
            cause = 'Traffic-driven CPU saturation, not DB exhaustion'
            summary = 'High incoming traffic is increasing CPU utilization while the database pool remains healthy.'
            action = 'Monitor application CPU and consider controlled horizontal scaling.'
            evidence.extend([f'CPU utilization: {cpu}%', f'Requests per minute: {requests}', f'Database connections: {db_connections}/{db_limit}'])
            reason = 'Internal model identified high traffic and CPU pressure.'
        else:
            cause = 'Unknown or insufficiently classified incident'
            summary = 'Available evidence does not match known incident rules.'
            action = 'Collect additional monitoring and log evidence.'
            evidence.append('No known incident pattern was identified.')
            reason = 'Insufficient evidence for a confident classification.'
        if previous:
            changes = [f"{key}: {previous['metrics'].get(key)} -> {value}" for key, value in metrics.items() if previous['metrics'].get(key) != value]
            evidence.append('Previous observation: incident state was ' + previous['incident_state'])
            evidence.append('Metric changes: ' + ('; '.join(changes) if changes else 'No metric changes detected.'))
        else:
            evidence.append('First observation: initial internal state created.')
        evidence.append(f'Internal observation number: {observation_count}')
        self.internal_state[sid] = {'observation_count': observation_count, 'incident_state': cause, 'metrics': dict(metrics), 'likely_cause': cause}
        return AgentResult(agent_name='Model-Based Reflex Agent', incident_summary=summary, evidence=tuple(evidence), likely_cause=cause, recommended_action=action, verification_steps=('Compare current and previous monitoring observations.', 'Inspect relevant application logs and deployment history.', 'Verify service recovery using health and availability metrics.'), limitations=('Uses a simple internal state rather than a learned model.', 'Depends on the quality of available logs and metrics.', 'Does not automatically execute recovery actions.'), reason=reason)

class GoalBasedAgent:

    def decide(self, scenario: IncidentScenario) -> AgentResult:
        metrics = scenario.metrics
        logs = list(scenario.logs)
        log_text = ' '.join(logs).lower()
        latency = metrics.get('latency_p95_ms', 0)
        error_rate = metrics.get('error_rate_pct', 0)
        cpu = metrics.get('cpu_pct', 0)
        requests = metrics.get('requests_per_min', 0)
        db_connections = metrics.get('db_connections_in_use', 0)
        db_limit = metrics.get('db_pool_limit', 0)
        availability = metrics.get('availability_pct', 0)
        if 'startup failed' in log_text or 'missing auth_issuer' in log_text:
            previous = dict(scenario.previous_version)
            deployments = [dict(item) for item in scenario.deployment_history]
            failed = next((item for item in reversed(deployments) if 'failed' in item.get('status', '').lower()), {})
            failed_version = failed.get('version', 'unknown')
            healthy_version = previous.get('version', 'unknown')
            cause = 'Application startup failure due to missing AUTH_ISSUER'
            summary = 'The application is unavailable because the latest deployment failed during startup.'
            action = f'Goal: Restore application availability. Step 1: Verify the missing AUTH_ISSUER setting. Step 2: Compare failed version {failed_version} with previous healthy version {healthy_version}. Step 3: After operator approval, restore the correct configuration or perform a controlled rollback. Step 4: Restart and verify application health.'
            goals = ('Restore application startup.', 'Restore service availability.', 'Reduce error rate to an acceptable level.')
            evidence = (f'Current availability: {availability}%', f'Current error rate: {error_rate}%', f'Failed deployment: {failed_version}', f'Previous healthy version: {healthy_version}', f"Previous configuration: {previous.get('config')}", 'Missing AUTH_ISSUER confirmed in application logs. The earlier DB transient timeout was marked resolved; it is historical and inconclusive for the current startup failure.')
            verification = ('Confirm the application starts successfully.', 'Verify the AUTH_ISSUER configuration.', 'Confirm availability has recovered.', 'Check that the error rate has decreased.')
            reason = 'Selected the application recovery goal because the service is unavailable.'
        elif db_limit > 0 and db_connections >= db_limit and (latency >= 1000 or error_rate >= 5):
            cause = 'Database connection pool exhaustion'
            summary = 'The database connection pool is exhausted, affecting API response time and availability.'
            action = 'Goal: Restore database connection availability. Step 1: Identify long-running or leaked connections. Step 2: Investigate connection wait queues. Step 3: Review safe pool capacity adjustments. Step 4: Verify database and API recovery.'
            goals = ('Bring database connections below the pool limit.', 'Reduce API latency.', 'Reduce application error rate.', 'Restore service availability.')
            evidence = (f'Database connections: {db_connections}/{db_limit}', f'P95 latency: {latency} ms', f'Error rate: {error_rate}%', f'Availability: {availability}%')
            verification = ('Confirm database connections are below the pool limit.', 'Check database connection wait queues.', 'Verify API latency has decreased.', 'Verify application errors have reduced.')
            reason = 'Selected database recovery goal because the connection pool is completely occupied.'
        elif cpu >= 85 and requests >= 4000 and (db_limit > 0) and (db_connections < db_limit * 0.8):
            cause = 'High traffic causing CPU saturation'
            summary = 'The system is experiencing high CPU utilization because of increased incoming traffic.'
            action = 'Goal: Reduce CPU pressure while maintaining service availability. Step 1: Identify traffic-heavy endpoints. Step 2: Review request handling and application workload. Step 3: Evaluate controlled horizontal scaling. Step 4: Monitor CPU and response times.'
            goals = ('Reduce CPU utilization below 80%.', 'Maintain healthy database connections.', 'Maintain low application error rates.', 'Preserve service availability.')
            evidence = (f'CPU utilization: {cpu}%', f'Requests per minute: {requests}', f'Database connections: {db_connections}/{db_limit}', f'Current error rate: {error_rate}%', f'Availability: {availability}%')
            verification = ('Confirm CPU utilization is decreasing.', 'Check that database connections remain healthy.', 'Verify API latency remains within the target.', 'Confirm application errors remain low.')
            reason = 'Selected CPU recovery goal because traffic is high and database capacity remains available.'
        else:
            cause = 'No critical incident pattern detected'
            summary = 'The available metrics do not indicate a known critical incident.'
            action = 'Goal: Maintain system health. Continue monitoring application metrics and logs.'
            goals = ('Maintain service availability.', 'Maintain acceptable response times.', 'Keep error rates low.')
            evidence = (f'CPU utilization: {cpu}%', f'Error rate: {error_rate}%', f'Availability: {availability}%')
            verification = ('Continue monitoring application health.', 'Review metrics for emerging problems.')
            reason = 'Normal monitoring goal selected.'
        return AgentResult(agent_name='Goal-Based Agent', incident_summary=summary, evidence=tuple(evidence) + ('Selected recovery goals: ' + '; '.join(goals),), likely_cause=cause, recommended_action=action, verification_steps=verification, limitations=('Uses predefined recovery goals.', 'Does not automatically execute recovery actions.', 'Depends on the quality of available metrics and logs.'), reason=reason)

class UtilityBasedAgent:

    def __init__(self):
        self.weights = {'benefit': 0.3, 'success': 0.25, 'speed': 0.2, 'safety': 0.15, 'cost_efficiency': 0.1}

    def calculate_utility(self, action):
        score = sum((self.weights[key] * action[key] for key in self.weights))
        return round(score, 2)

    def decide(self, scenario: IncidentScenario) -> AgentResult:
        metrics = scenario.metrics
        logs = list(scenario.logs)
        log_text = ' '.join(logs).lower()
        latency = metrics.get('latency_p95_ms', 0)
        error_rate = metrics.get('error_rate_pct', 0)
        cpu = metrics.get('cpu_pct', 0)
        requests = metrics.get('requests_per_min', 0)
        db_connections = metrics.get('db_connections_in_use', 0)
        db_limit = metrics.get('db_pool_limit', 0)
        availability = metrics.get('availability_pct', 0)
        if 'startup failed' in log_text or 'missing auth_issuer' in log_text:
            cause = 'Application startup failure due to missing AUTH_ISSUER'
            previous = dict(scenario.previous_version)
            deployments = [dict(item) for item in scenario.deployment_history]
            failed = next((item for item in reversed(deployments) if 'failed' in item.get('status', '').lower()), {})
            failed_version = failed.get('version', 'unknown')
            healthy_version = previous.get('version', 'unknown')
            actions = [{'name': 'Restore missing AUTH_ISSUER configuration', 'benefit': 10, 'success': 9, 'speed': 9, 'safety': 9, 'cost_efficiency': 10}, {'name': 'Controlled rollback to previous healthy version', 'benefit': 10, 'success': 9, 'speed': 8, 'safety': 8, 'cost_efficiency': 7}, {'name': 'Restart application without configuration correction', 'benefit': 2, 'success': 2, 'speed': 8, 'safety': 3, 'cost_efficiency': 9}]
            summary = f'Deployment {failed_version} failed because the AUTH_ISSUER setting is missing.'
            verification = ('Verify AUTH_ISSUER configuration.', 'Confirm application startup.', 'Check application availability.', 'Verify the error rate decreases.')
            evidence = [f'Failed version: {failed_version}', f'Previous healthy version: {healthy_version}', f"Previous configuration: {previous.get('config')}", f'Availability: {availability}%', f'Error rate: {error_rate}%', 'The earlier DB transient timeout was marked resolved; it is historical and inconclusive for the current startup failure.']
        elif db_limit > 0 and db_connections >= db_limit and (latency >= 1000 or error_rate >= 5):
            cause = 'Database connection pool exhaustion'
            actions = [{'name': 'Investigate and release stuck database connections', 'benefit': 10, 'success': 9, 'speed': 8, 'safety': 9, 'cost_efficiency': 9}, {'name': 'Increase database connection pool capacity after review', 'benefit': 8, 'success': 8, 'speed': 7, 'safety': 7, 'cost_efficiency': 7}, {'name': 'Restart all application services immediately', 'benefit': 6, 'success': 5, 'speed': 8, 'safety': 3, 'cost_efficiency': 6}]
            summary = 'The database connection pool is fully occupied, causing API delays and errors.'
            verification = ('Check database connection utilization.', 'Verify connection wait queues decrease.', 'Verify API latency improves.', 'Confirm application error rate decreases.')
            evidence = [f'Database connections: {db_connections}/{db_limit}', f'P95 latency: {latency} ms', f'Error rate: {error_rate}%', f'Availability: {availability}%']
        elif cpu >= 85 and requests >= 4000 and (db_limit > 0) and (db_connections < db_limit * 0.8):
            cause = 'High traffic causing CPU saturation'
            actions = [{'name': 'Controlled horizontal application scaling', 'benefit': 10, 'success': 9, 'speed': 8, 'safety': 9, 'cost_efficiency': 7}, {'name': 'Optimize CPU-intensive application requests', 'benefit': 8, 'success': 8, 'speed': 6, 'safety': 9, 'cost_efficiency': 9}, {'name': 'Immediately restart all services', 'benefit': 3, 'success': 3, 'speed': 7, 'safety': 2, 'cost_efficiency': 5}]
            summary = 'High incoming traffic is causing CPU saturation while database connections remain healthy.'
            verification = ('Monitor CPU utilization.', 'Check incoming request volume.', 'Confirm database connections remain healthy.', 'Verify API latency and error rate.')
            evidence = [f'CPU utilization: {cpu}%', f'Requests per minute: {requests}', f'Database connections: {db_connections}/{db_limit}', f'Error rate: {error_rate}%']
        else:
            cause = 'No known critical incident pattern detected'
            actions = [{'name': 'Continue monitoring system health', 'benefit': 9, 'success': 10, 'speed': 10, 'safety': 10, 'cost_efficiency': 10}, {'name': 'Perform unnecessary immediate restart', 'benefit': 2, 'success': 4, 'speed': 8, 'safety': 3, 'cost_efficiency': 4}]
            summary = 'No known critical incident was identified.'
            verification = ('Continue monitoring application metrics.', 'Review logs for new incidents.')
            evidence = [f'CPU utilization: {cpu}%', f'Error rate: {error_rate}%', f'Availability: {availability}%']
        for action in actions:
            action['utility_score'] = self.calculate_utility(action)
        best_action = max(actions, key=lambda action: action['utility_score'])
        action_scores = [f"{action['name']}: Utility score = {action['utility_score']}/10" for action in actions]
        evidence.extend(action_scores)
        evidence.append(f"Selected action: {best_action['name']}")
        evidence.append(f"Highest utility score: {best_action['utility_score']}/10")
        return AgentResult(agent_name='Utility-Based Agent', incident_summary=summary, evidence=tuple(evidence), likely_cause=cause, recommended_action=f"{best_action['name']}. Obtain operator approval before executing any production change.", verification_steps=verification, limitations=('Utility weights are predefined.', 'Scores are estimates and not measured recovery probabilities.', 'Does not automatically execute recovery actions.'), reason='Evaluated multiple recovery alternatives using weighted utility scores and selected the highest-scoring action.')
class LearningAgent:

    def __init__(self, experience_path=None):
        self.experience_path = Path(experience_path) if experience_path else PROJECT_ROOT / 'data' / 'learning_agent_experience.json'
        self.experience_path.parent.mkdir(parents=True, exist_ok=True)
        self.experience = {'schema_version': 1, 'feedback': {}}
        if self.experience_path.exists():
            self.experience = json.loads(self.experience_path.read_text(encoding='utf-8'))
        print('Learning Agent initialized.')
        print('Experience file:', self.experience_path)
        print('Existing feedback categories:', list(self.experience['feedback'].keys()))

    def _incident_details(self, scenario):
        metrics = scenario.metrics
        logs = ' '.join(scenario.logs).lower()
        latency = metrics.get('latency_p95_ms', 0)
        error = metrics.get('error_rate_pct', 0)
        cpu = metrics.get('cpu_pct', 0)
        requests = metrics.get('requests_per_min', 0)
        db = metrics.get('db_connections_in_use', 0)
        limit = metrics.get('db_pool_limit', 0)
        if 'startup failed' in logs or 'missing auth_issuer' in logs:
            category = 'configuration'
            cause = 'Startup failure due to missing AUTH_ISSUER'
            summary = 'Application startup failed after a configuration change.'
            previous = dict(scenario.previous_version)
            failed = next((dict(item) for item in reversed(scenario.deployment_history) if 'failed' in item.get('status', '').lower()), {})
            evidence = [f"Failed deployment: {failed.get('version')}", f"Previous healthy version: {previous.get('version')}", f"Previous configuration: {previous.get('config')}", 'The earlier DB transient timeout was marked resolved; it is historical and inconclusive for the current startup failure.']
            actions = [('Restore missing AUTH_ISSUER configuration', 9.4), ('Controlled rollback to previous healthy version', 8.75), ('Restart application without configuration correction', 4.05)]
            verification = ('Verify AUTH_ISSUER configuration.', 'Confirm successful application startup.', 'Check service availability and error rate.')
        elif limit > 0 and db >= limit and (latency >= 1000 or error >= 5):
            category = 'database'
            cause = 'Database connection pool exhaustion'
            summary = 'Database connections are fully occupied.'
            evidence = [f'Database connections: {db}/{limit}', f'P95 latency: {latency} ms', f'Error rate: {error}%']
            actions = [('Investigate and release stuck database connections', 9.1), ('Increase database connection pool capacity after review', 7.55), ('Restart all application services immediately', 5.7)]
            verification = ('Check database connection utilization.', 'Verify connection wait queues decrease.', 'Confirm API latency and errors improve.')
        elif cpu >= 85 and requests >= 4000 and (limit > 0) and (db < limit * 0.8):
            category = 'traffic'
            cause = 'High traffic causing CPU saturation'
            summary = 'High incoming traffic is increasing CPU utilization.'
            evidence = [f'CPU utilization: {cpu}%', f'Requests per minute: {requests}', f'Database connections: {db}/{limit}']
            actions = [('Controlled horizontal application scaling', 8.9), ('Optimize CPU-intensive application requests', 7.85), ('Immediately restart all services', 3.85)]
            verification = ('Monitor CPU utilization.', 'Check incoming traffic.', 'Verify database connections remain healthy.')
        else:
            category = 'uncertain'
            cause = 'Unknown incident'
            summary = 'Insufficient evidence to classify the incident.'
            evidence = ['Additional evidence is required.']
            actions = [('Collect additional monitoring evidence', 8.0)]
            verification = ('Collect additional metrics and logs.', 'Review the incident again.')
        return (category, cause, summary, evidence, actions, verification)

    def decide(self, scenario):
        category, cause, summary, evidence, actions, verification = self._incident_details(scenario)
        category_feedback = self.experience['feedback'].get(category, {})
        evaluated_actions = []
        for action_name, base_score in actions:
            record = category_feedback.get(action_name, {})
            successes = record.get('successes', 0)
            failures = record.get('failures', 0)
            attempts = successes + failures
            if attempts > 0:
                success_rate = successes / attempts
                learning_score = 0.7 * base_score + 0.3 * (success_rate * 10)
            else:
                learning_score = base_score
            evaluated_actions.append({'name': action_name, 'base_score': base_score, 'learning_score': round(learning_score, 2), 'feedback_count': attempts})
        best_action = max(evaluated_actions, key=lambda item: item['learning_score'])
        learning_evidence = list(evidence)
        for item in evaluated_actions:
            learning_evidence.append(f"{item['name']}: Base score={item['base_score']}, Learning score={item['learning_score']}, Feedback records={item['feedback_count']}")
        if any((item['feedback_count'] > 0 for item in evaluated_actions)):
            reason = 'Recommendation selected using predefined scores adjusted by previous feedback.'
        else:
            reason = 'No previous feedback available. Initial recommendation uses predefined scores.'
        learning_evidence.append(f"Selected action: {best_action['name']}")
        learning_evidence.append(f"Selected learning score: {best_action['learning_score']}/10")
        return AgentResult(agent_name='Learning Agent', incident_summary=summary, evidence=tuple(learning_evidence), likely_cause=cause, recommended_action=best_action['name'] + '. Obtain operator approval before production changes.', verification_steps=verification, limitations=('Learning depends on the quality of feedback.', 'Uses a simple feedback-based scoring mechanism.', 'Feedback scores do not guarantee real-world success.', 'Does not automatically execute recovery actions.'), reason=reason)

    def record_feedback(self, scenario, action_name, successful, notes=''):
        if not isinstance(successful, bool):
            raise ValueError('successful must be True or False')
        category, _, _, _, actions, _ = self._incident_details(scenario)
        valid_actions = [name for name, _ in actions]
        if action_name not in valid_actions:
            raise ValueError('Unknown action for this incident category.')
        category_feedback = self.experience['feedback'].setdefault(category, {})
        record = category_feedback.setdefault(action_name, {'successes': 0, 'failures': 0, 'notes': []})
        if successful:
            record['successes'] += 1
        else:
            record['failures'] += 1
        if notes:
            record['notes'].append(notes)
        self.experience_path.write_text(json.dumps(self.experience, indent=4), encoding='utf-8')
        print('Feedback successfully recorded!')
        print('Incident category:', category)
        print('Action:', action_name)
        print('Successful:', successful)
        print('Experience database updated:', self.experience_path)

    def available_actions(self, scenario: IncidentScenario) -> tuple[str, ...]:
        return tuple(name for name, _ in self._incident_details(scenario)[4])

def create_classical_agents(experience_path=None) -> dict[str, Agent]:
    """Create fresh instances of all five classical agents."""
    return {'Simple Reflex Agent': SimpleReflexAgent(), 'Model-Based Reflex Agent': ModelBasedReflexAgent(), 'Goal-Based Agent': GoalBasedAgent(), 'Utility-Based Agent': UtilityBasedAgent(), 'Learning Agent': LearningAgent(experience_path=experience_path if experience_path is not None else PROJECT_ROOT / 'data' / 'learning_agent_experience_integration.json')}
