"""QA-only removal of explicitly approved observation statements, never trading rules."""
import ast


def normalize_capture(tree, filename):
    class Normalize(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if filename=='live_runner.py' and node.name=='_learning_order_reference':return None
            if filename=='ml_engine.py':
                body=[];i=0
                while i<len(node.body):
                    n=node.body[i]
                    is_local=isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='audit_local' for t in n.targets)
                    if is_local and i+1<len(node.body):
                        following=node.body[i+1]
                        if isinstance(following,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='last_inference_snapshot' for t in following.targets):
                            body+=ast.parse('self._get_thread_local().last_inference_snapshot = snap').body
                            i+=5;continue
                    body.append(n);i+=1
                node.body=body
            return self.generic_visit(node)

        def visit_Assign(self,node):
            if filename=='ml_engine.py' and any(isinstance(t,ast.Name) and t.id=='audit_meta' for t in node.targets):return None
            if filename=='ml_engine.py' and any(
                isinstance(t,ast.Name) and t.id=='audit_local' or
                isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='audit_local'
                for t in node.targets):return None
            if filename=='moe_orchestrator.py' and any(isinstance(t,ast.Name) and t.id in {'learning_inference_completed','learning_snapshot'} for t in node.targets):return None
            return self.generic_visit(node)

        def visit_If(self,node):
            if filename=='ml_engine.py':
                mode=ast.parse('"mode" in target_live_df.columns',mode='eval').body
                quality=ast.parse('isinstance(audit_meta, dict) and not audit_meta.get("feature_complete", False)',mode='eval').body
                if ast.dump(node.test) in {ast.dump(mode),ast.dump(quality)}:return None
            if filename=='ml_engine.py' and isinstance(node.test,ast.Call) and isinstance(node.test.func,ast.Name) and node.test.func.id=='isinstance':
                if node.test.args and isinstance(node.test.args[0],ast.Attribute) and node.test.args[0].attr=='last_learning_snapshot':return None
            return self.generic_visit(node)

        def visit_Try(self,node):
            if filename=='ml_engine.py' and any(isinstance(n,ast.ImportFrom) and n.module=='core.learning_audit' for n in ast.walk(node)):return None
            if filename=='moe_orchestrator.py' and any(isinstance(n,ast.Name) and n.id=='captured' for n in ast.walk(node)):return None
            if filename=='live_runner.py' and any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in {
                'record_learning_execution','record_learning_decision','record_learning_cycle'} for n in ast.walk(node)):return None
            return self.generic_visit(node)

        def visit_Dict(self,node):
            keys={k.value for k in node.keys if isinstance(k,ast.Constant) and isinstance(k.value,str)}
            remove=set()
            if filename=='moe_orchestrator.py':remove={'learning_snapshot'}
            if filename=='live_runner.py':
                remove={'learning_snapshot_id','learning_feature_names'}
                if {'actual_entry_price','pnl_pct'}<=keys:
                    remove|={'active_model_sha256','active_model_id','active_model_schema','exit_price_source','fee_usd','execution_order_references','risk_settings'}
            pairs=[(k,v) for k,v in zip(node.keys,node.values) if not (isinstance(k,ast.Constant) and k.value in remove)]
            node.keys=[k for k,v in pairs];node.values=[v for k,v in pairs]
            return self.generic_visit(node)
    return Normalize().visit(tree)
