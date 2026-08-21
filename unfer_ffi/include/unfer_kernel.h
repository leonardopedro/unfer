/*
 * unfer_kernel.h — C ABI for the unfer probability kernel.
 *
 * GENERATED FILE — do not edit by hand. Regenerate with:
 *
 *     python3 gen_unfer_kernel_h.py
 *
 * All functions use i64-compatible parameters (ptr+len; ...) to match the CPS IR
 * calling convention. Return convention:
 *   >= 0 : success (handle, byte count, or 0)
 *   <  0 : error (-code); call uk_last_error() for a Diagnostic JSON.
 *
 * The `uz_*` declarations require building unfer_ffi with `--features zenodo`.
 */
#ifndef UNFER_KERNEL_H
#define UNFER_KERNEL_H

#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* ABI version (see unfer_protocol::KERNEL_VERSION). */
int64_t uk_version(void);
int64_t uk_action_apply(int64_t action_handle);
int64_t uk_action_get(int64_t action_handle,
              uint8_t* buf,
              int64_t cap);
int64_t uk_action_list(uint8_t* buf,
               int64_t cap);
int64_t uk_action_reject(int64_t action_handle);
int64_t uk_action_revert(int64_t action_handle);
int64_t uk_action_submit(const uint8_t* req_json,
                 int64_t len);
int64_t uk_agent_grants(int64_t handle,
                uint8_t* buf,
                int64_t cap);
int64_t uk_agent_kill(int64_t handle);
int64_t uk_agent_list(uint8_t* buf,
              int64_t cap);
int64_t uk_agent_spawn(const uint8_t* spec_json,
               int64_t len);
int64_t uk_auction_bid(const uint8_t* op_json,
               int64_t len);
int64_t uk_auction_close(const uint8_t* op_json,
                 int64_t len,
                 uint8_t* buf,
                 int64_t cap);
int64_t uk_auction_open(const uint8_t* op_json,
                int64_t len);
int64_t uk_auction_report(const uint8_t* lot_id_json,
                  int64_t len,
                  uint8_t* buf,
                  int64_t cap);
int64_t uk_audit_clear();
int64_t uk_audit_list(uint8_t* buf,
              int64_t cap);
int64_t uk_bayesian_update(int64_t model,
                   const uint8_t* req_json,
                   int64_t len);
int64_t uk_belief_propagation(int64_t model,
                      const uint8_t* req_json,
                      int64_t len);
int64_t uk_blueprint_cell(const uint8_t* id,
                  int64_t id_len,
                  uint8_t* buf,
                  int64_t cap);
int64_t uk_blueprint_export(int64_t model,
                    uint8_t* buf,
                    int64_t cap);
int64_t uk_blueprint_export_gadget(const uint8_t* id,
                           int64_t id_len,
                           uint8_t* buf,
                           int64_t cap);
int64_t uk_blueprint_get_by_id(const uint8_t* id,
                       int64_t id_len,
                       uint8_t* buf,
                       int64_t cap);
int64_t uk_blueprint_import(const uint8_t* cell,
                    int64_t len,
                    uint8_t* buf,
                    int64_t cap);
int64_t uk_blueprint_instantiate(const uint8_t* cell,
                         int64_t len);
int64_t uk_blueprint_list(uint8_t* buf,
                  int64_t cap);
int64_t uk_buf_free(int64_t handle);
int64_t uk_cert_burn(const uint8_t* op_json,
             int64_t len);
int64_t uk_cert_mint(const uint8_t* op_json,
             int64_t len);
int64_t uk_cert_mint_request(const uint8_t* req_json,
                     int64_t len);
int64_t uk_cert_root(uint8_t* buf,
             int64_t cap);
int64_t uk_cert_set_authority(const uint8_t* did_utf8,
                      int64_t len);
int64_t uk_cert_status(uint8_t* buf,
               int64_t cap);
int64_t uk_cert_transfer(const uint8_t* op_json,
                 int64_t len);
int64_t uk_condition(int64_t model,
             const uint8_t* event_json,
             int64_t len);
int64_t uk_event_probability(int64_t model,
                     const uint8_t* event_json,
                     int64_t len);
int64_t uk_evolve(int64_t model,
          const uint8_t* opts_json,
          int64_t len);
int64_t uk_gate_approve(int64_t action_handle);
int64_t uk_gate_list_pending(uint8_t* buf,
                     int64_t cap);
int64_t uk_gate_reject(int64_t action_handle);
int64_t uk_get_result(int64_t model,
              uint8_t* buf,
              int64_t cap);
int64_t uk_init(const uint8_t* _cfg_json,
        int64_t _len);
int64_t uk_last_error(uint8_t* buf,
              int64_t cap);
int64_t uk_logos_compile(int64_t model,
                 const uint8_t* sentence_ptr,
                 int64_t sentence_len);
int64_t uk_meter_status(const uint8_t* principal,
                int64_t len,
                const uint8_t* budget_json,
                int64_t blade,
                uint8_t* buf,
                int64_t cap);
int64_t uk_model_create(const uint8_t* spec_json,
                int64_t len);
int64_t uk_model_free(int64_t model);
int64_t uk_observability(const uint8_t* ptr,
                 int64_t len);
int64_t uk_observe(int64_t model,
           const uint8_t* obs_json,
           int64_t len);
int64_t uk_ode_analyze(const uint8_t* json,
               int64_t len);
double uk_ode_measure_original(int64_t model,
                        const uint8_t* var_json,
                        int64_t len);
int64_t uk_owner_clear();
int64_t uk_owner_list(uint8_t* buf,
              int64_t cap);
int64_t uk_owner_log(const uint8_t* ptr,
             int64_t len,
             const uint8_t* msg_ptr,
             int64_t msg_len);
int64_t uk_poll(int64_t sub,
        uint8_t* buf,
        int64_t cap);
int64_t uk_posture_get(uint8_t* buf,
               int64_t cap);
int64_t uk_posture_set(const uint8_t* posture_ptr,
               int64_t len);
int64_t uk_proof_verify(int64_t model,
                const uint8_t* export_ptr,
                int64_t export_len,
                const uint8_t* spec_json,
                int64_t spec_json_len);
int64_t uk_registry_vetted(const uint8_t* principal,
                   int64_t len,
                   int64_t vetted);
int64_t uk_report_issue(const uint8_t* ptr,
                int64_t len);
int64_t uk_request_resource(const uint8_t* resource_json,
                    int64_t len);
int64_t uk_resource_forfeit(const uint8_t* resource_json,
                    int64_t len);
int64_t uk_resource_introduce(const uint8_t* resource_json,
                      int64_t len);
int64_t uk_resource_pending(uint8_t* buf,
                    int64_t cap);
int64_t uk_resource_use(const uint8_t* resource_json,
                int64_t len);
int64_t uk_restore(const uint8_t* blob_json,
           int64_t len);
int64_t uk_secret_get(int64_t handle,
              const uint8_t* owner,
              int64_t owner_len,
              uint8_t* buf,
              int64_t cap);
int64_t uk_secret_put(const uint8_t* owner,
              int64_t owner_len,
              const uint8_t* value,
              int64_t value_len);
int64_t uk_secret_revoke(int64_t handle);
int64_t uk_session_compact(int64_t model,
                   int64_t seq);
int64_t uk_session_fork(int64_t model,
                int64_t seq);
int64_t uk_set_hamiltonian(int64_t model,
                   const uint8_t* json,
                   int64_t len);
int64_t uk_set_prior(int64_t model,
             const uint8_t* json,
             int64_t len);
int64_t uk_snapshot(int64_t model,
            uint8_t* buf,
            int64_t cap);
int64_t uk_subscribe(int64_t model,
             const uint8_t* query_json,
             int64_t len);
int64_t uk_symbolic_simplify(int64_t model,
                     const uint8_t* spec_json,
                     int64_t len);

int64_t uz_init(const uint8_t* cfg_json,
        int64_t cfg_len);
int64_t uz_last_error(uint8_t* buf,
              int64_t cap);
int64_t uz_manifest_json(uint8_t* buf,
                 int64_t cap);
int64_t uz_pull(uint8_t* buf,
        int64_t cap);
int64_t uz_push(const uint8_t* data,
        int64_t data_len,
        const uint8_t* frontier,
        int64_t frontier_len);

#ifdef __cplusplus
}
#endif

#endif /* UNFER_KERNEL_H */
