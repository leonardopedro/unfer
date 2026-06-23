/*
 * unfer_kernel.h — C ABI for the unfer probability kernel.
 *
 * All functions use i64-compatible parameters (ptr+len; scalar time goes
 * inside opts JSON) to match the CPS IR calling convention.
 *
 * Return convention:
 *   >= 0 : success (handle, byte count, or 0)
 *   <  0 : error (-code); call uk_last_error() for a Diagnostic JSON.
 *
 * Buffer protocol (uk_get_result, uk_last_error, uk_poll):
 *   Returns total bytes needed.  Copies min(needed, cap) into buf.
 *   If buf is NULL or cap <= 0, returns needed without copying.
 *   Caller re-calls with a buffer of at least `needed` bytes.
 *
 * Error codes are the UK-#### codes from unfer_protocol::codes.
 * See docs/PROTOCOL.md for the full code table.
 */
#ifndef UNFER_KERNEL_H
#define UNFER_KERNEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ABI version (currently 1). */
int64_t uk_version(void);

/* Initialize the kernel. cfg_json is optional ("{}" accepted). Returns 0. */
int64_t uk_init(const uint8_t* cfg_json, int64_t len);

/* Create a model session from a ModelSpec JSON. Returns positive handle. */
int64_t uk_model_create(const uint8_t* spec_json, int64_t len);

/* Free a model session. Returns 0 or -1004. */
int64_t uk_model_free(int64_t model);

/* Replace the prior state (PriorSpec JSON). Returns 0 or -code. */
int64_t uk_set_prior(int64_t model, const uint8_t* json, int64_t len);

/* Replace the Hamiltonian (HamiltonianSpec JSON). Returns 0 or -code. */
int64_t uk_set_hamiltonian(int64_t model, const uint8_t* json, int64_t len);

/* Evolve forward. opts_json = {"t": <seconds>}. Result via uk_get_result. */
int64_t uk_evolve(int64_t model, const uint8_t* opts_json, int64_t len);

/* Condition on an event (EventPredicate JSON). Result via uk_get_result. */
int64_t uk_condition(int64_t model, const uint8_t* event_json, int64_t len);

/* Compute P(event) without modifying state. Result via uk_get_result. */
int64_t uk_event_probability(int64_t model, const uint8_t* event_json, int64_t len);

/* Observe an event (v1: alias for uk_condition). */
int64_t uk_observe(int64_t model, const uint8_t* obs_json, int64_t len);

/* Retrieve last result JSON (buffer protocol). */
int64_t uk_get_result(int64_t model, uint8_t* buf, int64_t cap);

/* Retrieve last error as Diagnostic JSON (buffer protocol). */
int64_t uk_last_error(uint8_t* buf, int64_t cap);

/* Subscribe to a live event query. Returns positive sub handle. */
int64_t uk_subscribe(int64_t model, const uint8_t* query_json, int64_t len);

/* Poll a subscription (buffer protocol). Returns {"probability": <f64>}. */
int64_t uk_poll(int64_t sub, uint8_t* buf, int64_t cap);

#ifdef __cplusplus
}
#endif

#endif /* UNFER_KERNEL_H */
