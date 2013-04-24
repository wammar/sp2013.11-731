import basic_features
import alignment_features
import brown_clusters
import dwl_feature

FEATURES = [basic_features.n_oov, 
            basic_features.n_target,
            basic_features.n_target_type,
            basic_features.ef_ratio,
            basic_features.log_ef_ratio,
            alignment_features.dist_2_diag, 
            alignment_features.jump_dist, 
            alignment_features.fertilities,
            alignment_features.coarse_word_pair,
            brown_clusters.lm_score,
            brown_clusters.tm_score,
            dwl_feature.get_dwl_prob]
