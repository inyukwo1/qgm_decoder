# Generated from g4/SQLite.g4 by ANTLR 4.7.1
from antlr4 import *

if __name__ is not None and "." in __name__:
    from .SQLiteParser import SQLiteParser
else:
    from SQLiteParser import SQLiteParser

# This class defines a complete listener for a parse tree produced by SQLiteParser.
class SQLiteListener(ParseTreeListener):

    # Enter a parse tree produced by SQLiteParser#parse.
    def enterParse(self, ctx: SQLiteParser.ParseContext):
        pass

    # Exit a parse tree produced by SQLiteParser#parse.
    def exitParse(self, ctx: SQLiteParser.ParseContext):
        pass

    # Enter a parse tree produced by SQLiteParser#error.
    def enterError(self, ctx: SQLiteParser.ErrorContext):
        pass

    # Exit a parse tree produced by SQLiteParser#error.
    def exitError(self, ctx: SQLiteParser.ErrorContext):
        pass

    # Enter a parse tree produced by SQLiteParser#sql_stmt_list.
    def enterSql_stmt_list(self, ctx: SQLiteParser.Sql_stmt_listContext):
        pass

    # Exit a parse tree produced by SQLiteParser#sql_stmt_list.
    def exitSql_stmt_list(self, ctx: SQLiteParser.Sql_stmt_listContext):
        pass

    # Enter a parse tree produced by SQLiteParser#sql_stmt.
    def enterSql_stmt(self, ctx: SQLiteParser.Sql_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#sql_stmt.
    def exitSql_stmt(self, ctx: SQLiteParser.Sql_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#alter_table_stmt.
    def enterAlter_table_stmt(self, ctx: SQLiteParser.Alter_table_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#alter_table_stmt.
    def exitAlter_table_stmt(self, ctx: SQLiteParser.Alter_table_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#analyze_stmt.
    def enterAnalyze_stmt(self, ctx: SQLiteParser.Analyze_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#analyze_stmt.
    def exitAnalyze_stmt(self, ctx: SQLiteParser.Analyze_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#attach_stmt.
    def enterAttach_stmt(self, ctx: SQLiteParser.Attach_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#attach_stmt.
    def exitAttach_stmt(self, ctx: SQLiteParser.Attach_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#begin_stmt.
    def enterBegin_stmt(self, ctx: SQLiteParser.Begin_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#begin_stmt.
    def exitBegin_stmt(self, ctx: SQLiteParser.Begin_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#commit_stmt.
    def enterCommit_stmt(self, ctx: SQLiteParser.Commit_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#commit_stmt.
    def exitCommit_stmt(self, ctx: SQLiteParser.Commit_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#compound_select_stmt.
    def enterCompound_select_stmt(self, ctx: SQLiteParser.Compound_select_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#compound_select_stmt.
    def exitCompound_select_stmt(self, ctx: SQLiteParser.Compound_select_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#create_index_stmt.
    def enterCreate_index_stmt(self, ctx: SQLiteParser.Create_index_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#create_index_stmt.
    def exitCreate_index_stmt(self, ctx: SQLiteParser.Create_index_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#create_table_stmt.
    def enterCreate_table_stmt(self, ctx: SQLiteParser.Create_table_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#create_table_stmt.
    def exitCreate_table_stmt(self, ctx: SQLiteParser.Create_table_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#create_trigger_stmt.
    def enterCreate_trigger_stmt(self, ctx: SQLiteParser.Create_trigger_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#create_trigger_stmt.
    def exitCreate_trigger_stmt(self, ctx: SQLiteParser.Create_trigger_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#create_view_stmt.
    def enterCreate_view_stmt(self, ctx: SQLiteParser.Create_view_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#create_view_stmt.
    def exitCreate_view_stmt(self, ctx: SQLiteParser.Create_view_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#create_virtual_table_stmt.
    def enterCreate_virtual_table_stmt(
        self, ctx: SQLiteParser.Create_virtual_table_stmtContext
    ):
        pass

    # Exit a parse tree produced by SQLiteParser#create_virtual_table_stmt.
    def exitCreate_virtual_table_stmt(
        self, ctx: SQLiteParser.Create_virtual_table_stmtContext
    ):
        pass

    # Enter a parse tree produced by SQLiteParser#delete_stmt.
    def enterDelete_stmt(self, ctx: SQLiteParser.Delete_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#delete_stmt.
    def exitDelete_stmt(self, ctx: SQLiteParser.Delete_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#delete_stmt_limited.
    def enterDelete_stmt_limited(self, ctx: SQLiteParser.Delete_stmt_limitedContext):
        pass

    # Exit a parse tree produced by SQLiteParser#delete_stmt_limited.
    def exitDelete_stmt_limited(self, ctx: SQLiteParser.Delete_stmt_limitedContext):
        pass

    # Enter a parse tree produced by SQLiteParser#detach_stmt.
    def enterDetach_stmt(self, ctx: SQLiteParser.Detach_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#detach_stmt.
    def exitDetach_stmt(self, ctx: SQLiteParser.Detach_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#drop_index_stmt.
    def enterDrop_index_stmt(self, ctx: SQLiteParser.Drop_index_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#drop_index_stmt.
    def exitDrop_index_stmt(self, ctx: SQLiteParser.Drop_index_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#drop_table_stmt.
    def enterDrop_table_stmt(self, ctx: SQLiteParser.Drop_table_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#drop_table_stmt.
    def exitDrop_table_stmt(self, ctx: SQLiteParser.Drop_table_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#drop_trigger_stmt.
    def enterDrop_trigger_stmt(self, ctx: SQLiteParser.Drop_trigger_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#drop_trigger_stmt.
    def exitDrop_trigger_stmt(self, ctx: SQLiteParser.Drop_trigger_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#drop_view_stmt.
    def enterDrop_view_stmt(self, ctx: SQLiteParser.Drop_view_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#drop_view_stmt.
    def exitDrop_view_stmt(self, ctx: SQLiteParser.Drop_view_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#factored_select_stmt.
    def enterFactored_select_stmt(self, ctx: SQLiteParser.Factored_select_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#factored_select_stmt.
    def exitFactored_select_stmt(self, ctx: SQLiteParser.Factored_select_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#insert_stmt.
    def enterInsert_stmt(self, ctx: SQLiteParser.Insert_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#insert_stmt.
    def exitInsert_stmt(self, ctx: SQLiteParser.Insert_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#pragma_stmt.
    def enterPragma_stmt(self, ctx: SQLiteParser.Pragma_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#pragma_stmt.
    def exitPragma_stmt(self, ctx: SQLiteParser.Pragma_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#reindex_stmt.
    def enterReindex_stmt(self, ctx: SQLiteParser.Reindex_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#reindex_stmt.
    def exitReindex_stmt(self, ctx: SQLiteParser.Reindex_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#release_stmt.
    def enterRelease_stmt(self, ctx: SQLiteParser.Release_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#release_stmt.
    def exitRelease_stmt(self, ctx: SQLiteParser.Release_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#rollback_stmt.
    def enterRollback_stmt(self, ctx: SQLiteParser.Rollback_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#rollback_stmt.
    def exitRollback_stmt(self, ctx: SQLiteParser.Rollback_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#savepoint_stmt.
    def enterSavepoint_stmt(self, ctx: SQLiteParser.Savepoint_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#savepoint_stmt.
    def exitSavepoint_stmt(self, ctx: SQLiteParser.Savepoint_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#simple_select_stmt.
    def enterSimple_select_stmt(self, ctx: SQLiteParser.Simple_select_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#simple_select_stmt.
    def exitSimple_select_stmt(self, ctx: SQLiteParser.Simple_select_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#select_stmt.
    def enterSelect_stmt(self, ctx: SQLiteParser.Select_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#select_stmt.
    def exitSelect_stmt(self, ctx: SQLiteParser.Select_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#select_or_values.
    def enterSelect_or_values(self, ctx: SQLiteParser.Select_or_valuesContext):
        pass

    # Exit a parse tree produced by SQLiteParser#select_or_values.
    def exitSelect_or_values(self, ctx: SQLiteParser.Select_or_valuesContext):
        pass

    # Enter a parse tree produced by SQLiteParser#update_stmt.
    def enterUpdate_stmt(self, ctx: SQLiteParser.Update_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#update_stmt.
    def exitUpdate_stmt(self, ctx: SQLiteParser.Update_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#update_stmt_limited.
    def enterUpdate_stmt_limited(self, ctx: SQLiteParser.Update_stmt_limitedContext):
        pass

    # Exit a parse tree produced by SQLiteParser#update_stmt_limited.
    def exitUpdate_stmt_limited(self, ctx: SQLiteParser.Update_stmt_limitedContext):
        pass

    # Enter a parse tree produced by SQLiteParser#vacuum_stmt.
    def enterVacuum_stmt(self, ctx: SQLiteParser.Vacuum_stmtContext):
        pass

    # Exit a parse tree produced by SQLiteParser#vacuum_stmt.
    def exitVacuum_stmt(self, ctx: SQLiteParser.Vacuum_stmtContext):
        pass

    # Enter a parse tree produced by SQLiteParser#column_def.
    def enterColumn_def(self, ctx: SQLiteParser.Column_defContext):
        pass

    # Exit a parse tree produced by SQLiteParser#column_def.
    def exitColumn_def(self, ctx: SQLiteParser.Column_defContext):
        pass

    # Enter a parse tree produced by SQLiteParser#type_name.
    def enterType_name(self, ctx: SQLiteParser.Type_nameContext):
        pass

    # Exit a parse tree produced by SQLiteParser#type_name.
    def exitType_name(self, ctx: SQLiteParser.Type_nameContext):
        pass

    # Enter a parse tree produced by SQLiteParser#column_constraint.
    def enterColumn_constraint(self, ctx: SQLiteParser.Column_constraintContext):
        pass

    # Exit a parse tree produced by SQLiteParser#column_constraint.
    def exitColumn_constraint(self, ctx: SQLiteParser.Column_constraintContext):
        pass

    # Enter a parse tree produced by SQLiteParser#conflict_clause.
    def enterConflict_clause(self, ctx: SQLiteParser.Conflict_clauseContext):
        pass

    # Exit a parse tree produced by SQLiteParser#conflict_clause.
    def exitConflict_clause(self, ctx: SQLiteParser.Conflict_clauseContext):
        pass

    # Enter a parse tree produced by SQLiteParser#expr.
    def enterExpr(self, ctx: SQLiteParser.ExprContext):
        pass

    # Exit a parse tree produced by SQLiteParser#expr.
    def exitExpr(self, ctx: SQLiteParser.ExprContext):
        pass

    # Enter a parse tree produced by SQLiteParser#foreign_key_clause.
    def enterForeign_key_clause(self, ctx: SQLiteParser.Foreign_key_clauseContext):
        pass

    # Exit a parse tree produced by SQLiteParser#foreign_key_clause.
    def exitForeign_key_clause(self, ctx: SQLiteParser.Foreign_key_clauseContext):
        pass

    # Enter a parse tree produced by SQLiteParser#raise_function.
    def enterRaise_function(self, ctx: SQLiteParser.Raise_functionContext):
        pass

    # Exit a parse tree produced by SQLiteParser#raise_function.
    def exitRaise_function(self, ctx: SQLiteParser.Raise_functionContext):
        pass

    # Enter a parse tree produced by SQLiteParser#indexed_column.
    def enterIndexed_column(self, ctx: SQLiteParser.Indexed_columnContext):
        pass

    # Exit a parse tree produced by SQLiteParser#indexed_column.
    def exitIndexed_column(self, ctx: SQLiteParser.Indexed_columnContext):
        pass

    # Enter a parse tree produced by SQLiteParser#table_constraint.
    def enterTable_constraint(self, ctx: SQLiteParser.Table_constraintContext):
        pass

    # Exit a parse tree produced by SQLiteParser#table_constraint.
    def exitTable_constraint(self, ctx: SQLiteParser.Table_constraintContext):
        pass

    # Enter a parse tree produced by SQLiteParser#with_clause.
    def enterWith_clause(self, ctx: SQLiteParser.With_clauseContext):
        pass

    # Exit a parse tree produced by SQLiteParser#with_clause.
    def exitWith_clause(self, ctx: SQLiteParser.With_clauseContext):
        pass

    # Enter a parse tree produced by SQLiteParser#qualified_table_name.
    def enterQualified_table_name(self, ctx: SQLiteParser.Qualified_table_nameContext):
        pass

    # Exit a parse tree produced by SQLiteParser#qualified_table_name.
    def exitQualified_table_name(self, ctx: SQLiteParser.Qualified_table_nameContext):
        pass

    # Enter a parse tree produced by SQLiteParser#ordering_term.
    def enterOrdering_term(self, ctx: SQLiteParser.Ordering_termContext):
        pass

    # Exit a parse tree produced by SQLiteParser#ordering_term.
    def exitOrdering_term(self, ctx: SQLiteParser.Ordering_termContext):
        pass

    # Enter a parse tree produced by SQLiteParser#pragma_value.
    def enterPragma_value(self, ctx: SQLiteParser.Pragma_valueContext):
        pass

    # Exit a parse tree produced by SQLiteParser#pragma_value.
    def exitPragma_value(self, ctx: SQLiteParser.Pragma_valueContext):
        pass

    # Enter a parse tree produced by SQLiteParser#common_table_expression.
    def enterCommon_table_expression(
        self, ctx: SQLiteParser.Common_table_expressionContext
    ):
        pass

    # Exit a parse tree produced by SQLiteParser#common_table_expression.
    def exitCommon_table_expression(
        self, ctx: SQLiteParser.Common_table_expressionContext
    ):
        pass

    # Enter a parse tree produced by SQLiteParser#result_column.
    def enterResult_column(self, ctx: SQLiteParser.Result_columnContext):
        pass

    # Exit a parse tree produced by SQLiteParser#result_column.
    def exitResult_column(self, ctx: SQLiteParser.Result_columnContext):
        pass

    # Enter a parse tree produced by SQLiteParser#table_or_subquery.
    def enterTable_or_subquery(self, ctx: SQLiteParser.Table_or_subqueryContext):
        pass

    # Exit a parse tree produced by SQLiteParser#table_or_subquery.
    def exitTable_or_subquery(self, ctx: SQLiteParser.Table_or_subqueryContext):
        pass

    # Enter a parse tree produced by SQLiteParser#join_clause.
    def enterJoin_clause(self, ctx: SQLiteParser.Join_clauseContext):
        pass

    # Exit a parse tree produced by SQLiteParser#join_clause.
    def exitJoin_clause(self, ctx: SQLiteParser.Join_clauseContext):
        pass

    # Enter a parse tree produced by SQLiteParser#join_operator.
    def enterJoin_operator(self, ctx: SQLiteParser.Join_operatorContext):
        pass

    # Exit a parse tree produced by SQLiteParser#join_operator.
    def exitJoin_operator(self, ctx: SQLiteParser.Join_operatorContext):
        pass

    # Enter a parse tree produced by SQLiteParser#join_constraint.
    def enterJoin_constraint(self, ctx: SQLiteParser.Join_constraintContext):
        pass

    # Exit a parse tree produced by SQLiteParser#join_constraint.
    def exitJoin_constraint(self, ctx: SQLiteParser.Join_constraintContext):
        pass

    # Enter a parse tree produced by SQLiteParser#select_core.
    def enterSelect_core(self, ctx: SQLiteParser.Select_coreContext):
        pass

    # Exit a parse tree produced by SQLiteParser#select_core.
    def exitSelect_core(self, ctx: SQLiteParser.Select_coreContext):
        pass

    # Enter a parse tree produced by SQLiteParser#compound_operator.
    def enterCompound_operator(self, ctx: SQLiteParser.Compound_operatorContext):
        pass

    # Exit a parse tree produced by SQLiteParser#compound_operator.
    def exitCompound_operator(self, ctx: SQLiteParser.Compound_operatorContext):
        pass

    # Enter a parse tree produced by SQLiteParser#cte_table_name.
    def enterCte_table_name(self, ctx: SQLiteParser.Cte_table_nameContext):
        pass

    # Exit a parse tree produced by SQLiteParser#cte_table_name.
    def exitCte_table_name(self, ctx: SQLiteParser.Cte_table_nameContext):
        pass

    # Enter a parse tree produced by SQLiteParser#signed_number.
    def enterSigned_number(self, ctx: SQLiteParser.Signed_numberContext):
        pass

    # Exit a parse tree produced by SQLiteParser#signed_number.
    def exitSigned_number(self, ctx: SQLiteParser.Signed_numberContext):
        pass

    # Enter a parse tree produced by SQLiteParser#literal_value.
    def enterLiteral_value(self, ctx: SQLiteParser.Literal_valueContext):
        pass

    # Exit a parse tree produced by SQLiteParser#literal_value.
    def exitLiteral_value(self, ctx: SQLiteParser.Literal_valueContext):
        pass

    # Enter a parse tree produced by SQLiteParser#unary_operator.
    def enterUnary_operator(self, ctx: SQLiteParser.Unary_operatorContext):
        pass

    # Exit a parse tree produced by SQLiteParser#unary_operator.
    def exitUnary_operator(self, ctx: SQLiteParser.Unary_operatorContext):
        pass

    # Enter a parse tree produced by SQLiteParser#error_message.
    def enterError_message(self, ctx: SQLiteParser.Error_messageContext):
        pass

    # Exit a parse tree produced by SQLiteParser#error_message.
    def exitError_message(self, ctx: SQLiteParser.Error_messageContext):
        pass

    # Enter a parse tree produced by SQLiteParser#module_argument.
    def enterModule_argument(self, ctx: SQLiteParser.Module_argumentContext):
        pass

    # Exit a parse tree produced by SQLiteParser#module_argument.
    def exitModule_argument(self, ctx: SQLiteParser.Module_argumentContext):
        pass

    # Enter a parse tree produced by SQLiteParser#column_alias.
    def enterColumn_alias(self, ctx: SQLiteParser.Column_aliasContext):
        pass

    # Exit a parse tree produced by SQLiteParser#column_alias.
    def exitColumn_alias(self, ctx: SQLiteParser.Column_aliasContext):
        pass

    # Enter a parse tree produced by SQLiteParser#select_clause.
    def enterSelect_clause(self, ctx: SQLiteParser.Select_clauseContext):
        pass

    # Exit a parse tree produced by SQLiteParser#select_clause.
    def exitSelect_clause(self, ctx: SQLiteParser.Select_clauseContext):
        pass

    # Enter a parse tree produced by SQLiteParser#from_clause.
    def enterFrom_clause(self, ctx: SQLiteParser.From_clauseContext):
        pass

    # Exit a parse tree produced by SQLiteParser#from_clause.
    def exitFrom_clause(self, ctx: SQLiteParser.From_clauseContext):
        pass

    # Enter a parse tree produced by SQLiteParser#having_clause.
    def enterHaving_clause(self, ctx: SQLiteParser.Having_clauseContext):
        pass

    # Exit a parse tree produced by SQLiteParser#having_clause.
    def exitHaving_clause(self, ctx: SQLiteParser.Having_clauseContext):
        pass

    # Enter a parse tree produced by SQLiteParser#group_clause.
    def enterGroup_clause(self, ctx: SQLiteParser.Group_clauseContext):
        pass

    # Exit a parse tree produced by SQLiteParser#group_clause.
    def exitGroup_clause(self, ctx: SQLiteParser.Group_clauseContext):
        pass

    # Enter a parse tree produced by SQLiteParser#where_clause.
    def enterWhere_clause(self, ctx: SQLiteParser.Where_clauseContext):
        pass

    # Exit a parse tree produced by SQLiteParser#where_clause.
    def exitWhere_clause(self, ctx: SQLiteParser.Where_clauseContext):
        pass

    # Enter a parse tree produced by SQLiteParser#order_clause.
    def enterOrder_clause(self, ctx: SQLiteParser.Order_clauseContext):
        pass

    # Exit a parse tree produced by SQLiteParser#order_clause.
    def exitOrder_clause(self, ctx: SQLiteParser.Order_clauseContext):
        pass

    # Enter a parse tree produced by SQLiteParser#limit_clause.
    def enterLimit_clause(self, ctx: SQLiteParser.Limit_clauseContext):
        pass

    # Exit a parse tree produced by SQLiteParser#limit_clause.
    def exitLimit_clause(self, ctx: SQLiteParser.Limit_clauseContext):
        pass

    # Enter a parse tree produced by SQLiteParser#like_clause.
    def enterLike_clause(self, ctx: SQLiteParser.Like_clauseContext):
        pass

    # Exit a parse tree produced by SQLiteParser#like_clause.
    def exitLike_clause(self, ctx: SQLiteParser.Like_clauseContext):
        pass

    # Enter a parse tree produced by SQLiteParser#case_clause.
    def enterCase_clause(self, ctx: SQLiteParser.Case_clauseContext):
        pass

    # Exit a parse tree produced by SQLiteParser#case_clause.
    def exitCase_clause(self, ctx: SQLiteParser.Case_clauseContext):
        pass

    # Enter a parse tree produced by SQLiteParser#binary_operator.
    def enterBinary_operator(self, ctx: SQLiteParser.Binary_operatorContext):
        pass

    # Exit a parse tree produced by SQLiteParser#binary_operator.
    def exitBinary_operator(self, ctx: SQLiteParser.Binary_operatorContext):
        pass

    # Enter a parse tree produced by SQLiteParser#k_distinct.
    def enterK_distinct(self, ctx: SQLiteParser.K_distinctContext):
        pass

    # Exit a parse tree produced by SQLiteParser#k_distinct.
    def exitK_distinct(self, ctx: SQLiteParser.K_distinctContext):
        pass

    # Enter a parse tree produced by SQLiteParser#k_all.
    def enterK_all(self, ctx: SQLiteParser.K_allContext):
        pass

    # Exit a parse tree produced by SQLiteParser#k_all.
    def exitK_all(self, ctx: SQLiteParser.K_allContext):
        pass

    # Enter a parse tree produced by SQLiteParser#k_and.
    def enterK_and(self, ctx: SQLiteParser.K_andContext):
        pass

    # Exit a parse tree produced by SQLiteParser#k_and.
    def exitK_and(self, ctx: SQLiteParser.K_andContext):
        pass

    # Enter a parse tree produced by SQLiteParser#k_or.
    def enterK_or(self, ctx: SQLiteParser.K_orContext):
        pass

    # Exit a parse tree produced by SQLiteParser#k_or.
    def exitK_or(self, ctx: SQLiteParser.K_orContext):
        pass

    # Enter a parse tree produced by SQLiteParser#k_asc.
    def enterK_asc(self, ctx: SQLiteParser.K_ascContext):
        pass

    # Exit a parse tree produced by SQLiteParser#k_asc.
    def exitK_asc(self, ctx: SQLiteParser.K_ascContext):
        pass

    # Enter a parse tree produced by SQLiteParser#k_between.
    def enterK_between(self, ctx: SQLiteParser.K_betweenContext):
        pass

    # Exit a parse tree produced by SQLiteParser#k_between.
    def exitK_between(self, ctx: SQLiteParser.K_betweenContext):
        pass

    # Enter a parse tree produced by SQLiteParser#k_natural.
    def enterK_natural(self, ctx: SQLiteParser.K_naturalContext):
        pass

    # Exit a parse tree produced by SQLiteParser#k_natural.
    def exitK_natural(self, ctx: SQLiteParser.K_naturalContext):
        pass

    # Enter a parse tree produced by SQLiteParser#k_left.
    def enterK_left(self, ctx: SQLiteParser.K_leftContext):
        pass

    # Exit a parse tree produced by SQLiteParser#k_left.
    def exitK_left(self, ctx: SQLiteParser.K_leftContext):
        pass

    # Enter a parse tree produced by SQLiteParser#k_outer.
    def enterK_outer(self, ctx: SQLiteParser.K_outerContext):
        pass

    # Exit a parse tree produced by SQLiteParser#k_outer.
    def exitK_outer(self, ctx: SQLiteParser.K_outerContext):
        pass

    # Enter a parse tree produced by SQLiteParser#k_inner.
    def enterK_inner(self, ctx: SQLiteParser.K_innerContext):
        pass

    # Exit a parse tree produced by SQLiteParser#k_inner.
    def exitK_inner(self, ctx: SQLiteParser.K_innerContext):
        pass

    # Enter a parse tree produced by SQLiteParser#k_cross.
    def enterK_cross(self, ctx: SQLiteParser.K_crossContext):
        pass

    # Exit a parse tree produced by SQLiteParser#k_cross.
    def exitK_cross(self, ctx: SQLiteParser.K_crossContext):
        pass

    # Enter a parse tree produced by SQLiteParser#k_join.
    def enterK_join(self, ctx: SQLiteParser.K_joinContext):
        pass

    # Exit a parse tree produced by SQLiteParser#k_join.
    def exitK_join(self, ctx: SQLiteParser.K_joinContext):
        pass

    # Enter a parse tree produced by SQLiteParser#k_current_date.
    def enterK_current_date(self, ctx: SQLiteParser.K_current_dateContext):
        pass

    # Exit a parse tree produced by SQLiteParser#k_current_date.
    def exitK_current_date(self, ctx: SQLiteParser.K_current_dateContext):
        pass

    # Enter a parse tree produced by SQLiteParser#k_current_time.
    def enterK_current_time(self, ctx: SQLiteParser.K_current_timeContext):
        pass

    # Exit a parse tree produced by SQLiteParser#k_current_time.
    def exitK_current_time(self, ctx: SQLiteParser.K_current_timeContext):
        pass

    # Enter a parse tree produced by SQLiteParser#k_current_timestamp.
    def enterK_current_timestamp(self, ctx: SQLiteParser.K_current_timestampContext):
        pass

    # Exit a parse tree produced by SQLiteParser#k_current_timestamp.
    def exitK_current_timestamp(self, ctx: SQLiteParser.K_current_timestampContext):
        pass

    # Enter a parse tree produced by SQLiteParser#k_null.
    def enterK_null(self, ctx: SQLiteParser.K_nullContext):
        pass

    # Exit a parse tree produced by SQLiteParser#k_null.
    def exitK_null(self, ctx: SQLiteParser.K_nullContext):
        pass

    # Enter a parse tree produced by SQLiteParser#numeric_literal.
    def enterNumeric_literal(self, ctx: SQLiteParser.Numeric_literalContext):
        pass

    # Exit a parse tree produced by SQLiteParser#numeric_literal.
    def exitNumeric_literal(self, ctx: SQLiteParser.Numeric_literalContext):
        pass

    # Enter a parse tree produced by SQLiteParser#string_literal.
    def enterString_literal(self, ctx: SQLiteParser.String_literalContext):
        pass

    # Exit a parse tree produced by SQLiteParser#string_literal.
    def exitString_literal(self, ctx: SQLiteParser.String_literalContext):
        pass

    # Enter a parse tree produced by SQLiteParser#blob_literal.
    def enterBlob_literal(self, ctx: SQLiteParser.Blob_literalContext):
        pass

    # Exit a parse tree produced by SQLiteParser#blob_literal.
    def exitBlob_literal(self, ctx: SQLiteParser.Blob_literalContext):
        pass

    # Enter a parse tree produced by SQLiteParser#k_desc.
    def enterK_desc(self, ctx: SQLiteParser.K_descContext):
        pass

    # Exit a parse tree produced by SQLiteParser#k_desc.
    def exitK_desc(self, ctx: SQLiteParser.K_descContext):
        pass

    # Enter a parse tree produced by SQLiteParser#k_is.
    def enterK_is(self, ctx: SQLiteParser.K_isContext):
        pass

    # Exit a parse tree produced by SQLiteParser#k_is.
    def exitK_is(self, ctx: SQLiteParser.K_isContext):
        pass

    # Enter a parse tree produced by SQLiteParser#k_in.
    def enterK_in(self, ctx: SQLiteParser.K_inContext):
        pass

    # Exit a parse tree produced by SQLiteParser#k_in.
    def exitK_in(self, ctx: SQLiteParser.K_inContext):
        pass

    # Enter a parse tree produced by SQLiteParser#k_not.
    def enterK_not(self, ctx: SQLiteParser.K_notContext):
        pass

    # Exit a parse tree produced by SQLiteParser#k_not.
    def exitK_not(self, ctx: SQLiteParser.K_notContext):
        pass

    # Enter a parse tree produced by SQLiteParser#k_exists.
    def enterK_exists(self, ctx: SQLiteParser.K_existsContext):
        pass

    # Exit a parse tree produced by SQLiteParser#k_exists.
    def exitK_exists(self, ctx: SQLiteParser.K_existsContext):
        pass

    # Enter a parse tree produced by SQLiteParser#keyword.
    def enterKeyword(self, ctx: SQLiteParser.KeywordContext):
        pass

    # Exit a parse tree produced by SQLiteParser#keyword.
    def exitKeyword(self, ctx: SQLiteParser.KeywordContext):
        pass

    # Enter a parse tree produced by SQLiteParser#name.
    def enterName(self, ctx: SQLiteParser.NameContext):
        pass

    # Exit a parse tree produced by SQLiteParser#name.
    def exitName(self, ctx: SQLiteParser.NameContext):
        pass

    # Enter a parse tree produced by SQLiteParser#function_name.
    def enterFunction_name(self, ctx: SQLiteParser.Function_nameContext):
        pass

    # Exit a parse tree produced by SQLiteParser#function_name.
    def exitFunction_name(self, ctx: SQLiteParser.Function_nameContext):
        pass

    # Enter a parse tree produced by SQLiteParser#database_name.
    def enterDatabase_name(self, ctx: SQLiteParser.Database_nameContext):
        pass

    # Exit a parse tree produced by SQLiteParser#database_name.
    def exitDatabase_name(self, ctx: SQLiteParser.Database_nameContext):
        pass

    # Enter a parse tree produced by SQLiteParser#table_name.
    def enterTable_name(self, ctx: SQLiteParser.Table_nameContext):
        pass

    # Exit a parse tree produced by SQLiteParser#table_name.
    def exitTable_name(self, ctx: SQLiteParser.Table_nameContext):
        pass

    # Enter a parse tree produced by SQLiteParser#table_or_index_name.
    def enterTable_or_index_name(self, ctx: SQLiteParser.Table_or_index_nameContext):
        pass

    # Exit a parse tree produced by SQLiteParser#table_or_index_name.
    def exitTable_or_index_name(self, ctx: SQLiteParser.Table_or_index_nameContext):
        pass

    # Enter a parse tree produced by SQLiteParser#new_table_name.
    def enterNew_table_name(self, ctx: SQLiteParser.New_table_nameContext):
        pass

    # Exit a parse tree produced by SQLiteParser#new_table_name.
    def exitNew_table_name(self, ctx: SQLiteParser.New_table_nameContext):
        pass

    # Enter a parse tree produced by SQLiteParser#column_name.
    def enterColumn_name(self, ctx: SQLiteParser.Column_nameContext):
        pass

    # Exit a parse tree produced by SQLiteParser#column_name.
    def exitColumn_name(self, ctx: SQLiteParser.Column_nameContext):
        pass

    # Enter a parse tree produced by SQLiteParser#collation_name.
    def enterCollation_name(self, ctx: SQLiteParser.Collation_nameContext):
        pass

    # Exit a parse tree produced by SQLiteParser#collation_name.
    def exitCollation_name(self, ctx: SQLiteParser.Collation_nameContext):
        pass

    # Enter a parse tree produced by SQLiteParser#foreign_table.
    def enterForeign_table(self, ctx: SQLiteParser.Foreign_tableContext):
        pass

    # Exit a parse tree produced by SQLiteParser#foreign_table.
    def exitForeign_table(self, ctx: SQLiteParser.Foreign_tableContext):
        pass

    # Enter a parse tree produced by SQLiteParser#index_name.
    def enterIndex_name(self, ctx: SQLiteParser.Index_nameContext):
        pass

    # Exit a parse tree produced by SQLiteParser#index_name.
    def exitIndex_name(self, ctx: SQLiteParser.Index_nameContext):
        pass

    # Enter a parse tree produced by SQLiteParser#trigger_name.
    def enterTrigger_name(self, ctx: SQLiteParser.Trigger_nameContext):
        pass

    # Exit a parse tree produced by SQLiteParser#trigger_name.
    def exitTrigger_name(self, ctx: SQLiteParser.Trigger_nameContext):
        pass

    # Enter a parse tree produced by SQLiteParser#view_name.
    def enterView_name(self, ctx: SQLiteParser.View_nameContext):
        pass

    # Exit a parse tree produced by SQLiteParser#view_name.
    def exitView_name(self, ctx: SQLiteParser.View_nameContext):
        pass

    # Enter a parse tree produced by SQLiteParser#module_name.
    def enterModule_name(self, ctx: SQLiteParser.Module_nameContext):
        pass

    # Exit a parse tree produced by SQLiteParser#module_name.
    def exitModule_name(self, ctx: SQLiteParser.Module_nameContext):
        pass

    # Enter a parse tree produced by SQLiteParser#pragma_name.
    def enterPragma_name(self, ctx: SQLiteParser.Pragma_nameContext):
        pass

    # Exit a parse tree produced by SQLiteParser#pragma_name.
    def exitPragma_name(self, ctx: SQLiteParser.Pragma_nameContext):
        pass

    # Enter a parse tree produced by SQLiteParser#savepoint_name.
    def enterSavepoint_name(self, ctx: SQLiteParser.Savepoint_nameContext):
        pass

    # Exit a parse tree produced by SQLiteParser#savepoint_name.
    def exitSavepoint_name(self, ctx: SQLiteParser.Savepoint_nameContext):
        pass

    # Enter a parse tree produced by SQLiteParser#table_alias.
    def enterTable_alias(self, ctx: SQLiteParser.Table_aliasContext):
        pass

    # Exit a parse tree produced by SQLiteParser#table_alias.
    def exitTable_alias(self, ctx: SQLiteParser.Table_aliasContext):
        pass

    # Enter a parse tree produced by SQLiteParser#transaction_name.
    def enterTransaction_name(self, ctx: SQLiteParser.Transaction_nameContext):
        pass

    # Exit a parse tree produced by SQLiteParser#transaction_name.
    def exitTransaction_name(self, ctx: SQLiteParser.Transaction_nameContext):
        pass

    # Enter a parse tree produced by SQLiteParser#any_name.
    def enterAny_name(self, ctx: SQLiteParser.Any_nameContext):
        pass

    # Exit a parse tree produced by SQLiteParser#any_name.
    def exitAny_name(self, ctx: SQLiteParser.Any_nameContext):
        pass
