#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/core/session_log.hpp"

#include <memory>
#include <string>

namespace strata::core {

// Caller synchronizes the device before saving, and after restoring all layers.
// include_index is false for the draft layer (its attention has no indexer).
size_t conversation_kv_bytes(const QsaState& state, const ModelGeometry& g, int64_t upto, bool include_index);
// A nonzero unchanged_tokens is valid only for storage retained from an image
// actually restored into this session, bounded by every subsequent rewrite.
// Equal token IDs alone do not establish that its K/V bytes are unchanged.
bool conversation_kv_save(ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                          int64_t upto, bool include_index, std::string& error,
                          int64_t unchanged_tokens = 0, size_t* reused_bytes = nullptr);
bool conversation_kv_capture_bytes(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                                   int64_t upto, bool include_index, size_t& bytes, std::string& error);
// No CUDA calls or destination writes. Used for whole-session prevalidation.
bool conversation_kv_validate(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                              int64_t upto, bool include_index, std::string& error);
// The read-stall split (output only): what the snapshot save/restore calls on this thread spent their time in -
// buffer growth (alloc), the copies, the device syncs they wait behind, and the checkpoints (the tree's host
// copies and the live session's running state).  Takes and resets the thread's record: call it once after a whole
// park or restore, and log what it returns.  `total_ms` >= 0 also appends the part of the caller's whole
// operation the split did not cover ("wait N ms", PLAN-p124a), so one line times the park end to end.
std::string conversation_snapshot_take_split(double total_ms = -1.0);
// Discard whatever the thread's record holds without printing it (PLAN-p124a): the park and the restore take it
// at their end; the steps that print no split (a turn save's syncs, a borrow's copies) must not add their time
// to the next park's, or the split can exceed the whole it claims to break down.
void conversation_snapshot_reset_split();
bool conversation_kv_restore(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                             int64_t upto, bool include_index, std::string& error);
// conversation_kv_save(from) + conversation_kv_restore(to) without a host image, for two states of the same layout.
// The copies are complete on return; the residency refresh is queued on the legacy stream like the restore's.
// `queued`: the copies are only queued on the private copy stream, the refresh is ordered after them on the device,
// and the caller synchronizes the device before it reads or writes either state.
bool conversation_kv_copy(const QsaState& from, const QsaState& to, const ModelGeometry& g,
                          int64_t upto, bool include_index, std::string& error, bool queued = false);
// The first `upto` tokens of an image captured at `captured` tokens: the leading bytes of each buffer (cells come
// first in the identity layout, as conversation_kv_save takes them), the image untouched.  Cells and indexer rows
// past `upto` in the restored pages hold the image's later tokens until a request writes them, as after a whole
// restore followed by a rewind to a checkpoint.
bool conversation_kv_restore_prefix(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                                    int64_t captured, int64_t upto, bool include_index, std::string& error);
// Diagnostic read-back after a synchronized restore. Uses 64 KiB of stack
// workspace, compares authoritative bytes and resident draft-ring pages, and
// fingerprints the authoritative payload only. Never changes model state.
bool conversation_kv_verify(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                            int64_t upto, bool include_index, uint64_t& fingerprint, std::string& error);
// Disk save without a host copy: the layer's authoritative K/V as a streamed source (same bytes and metadata as
// conversation_kv_save).  Caller synchronizes the device first and keeps the state untouched while it is read.
bool conversation_kv_source(SessionKvSource& source, const QsaState& state, const ModelGeometry& g,
                            int64_t upto, bool include_index, std::string& error);
// The bytes of each K/V part (k, v, k_scale, v_scale, pooled) a snapshot of `upto` tokens holds: no state read.
bool conversation_kv_part_sizes(const QsaState& state, const ModelGeometry& g, int64_t upto, bool include_index,
                                std::array<uint64_t, 5>& sizes, std::string& error);
// Disk sessions: the read limits this session can ever restore - its geometry and layer range, at most
// min(max_tokens, its cells) tokens, `max_checkpoints` checkpoints, the exact running-state sizes and the K/V part
// sizes at that many tokens - so session_file_read refuses an oversized or foreign file before it allocates.
bool conversation_session_read_limits(SessionReadLimits& limits, const SessionState& session, const ModelGeometry& g,
                                      const QsaState& draft, uint64_t max_tokens, uint64_t max_checkpoints,
                                      std::string& error);

struct ConversationStateSizes {
    size_t gdn = 0, ple = 0, tail = 0, dead = 0, block_pos = 0;
};
/// Whole-model sizes: `gdn` covers every GDN layer, the indexer sizes are per QSA layer.
bool conversation_state_sizes(const ModelGeometry& g, ConversationStateSizes& sizes, std::string& error);
/// The same for one session's layer carve (#216): `gdn` covers its `gdn_alloc` rows; the per-QSA-layer sizes
/// apply to each owned state [qsa_ord0, qsa_ord0 + qsa_alloc).  Rejects an inconsistent carve.  Checkpoints,
/// snapshots and their validation all use this: a session saves and restores only the state it owns.
bool conversation_session_sizes(const ModelGeometry& g, const SessionState& session, ConversationStateSizes& sizes,
                                std::string& error);
bool conversation_checkpoint_validate(const ConversationCheckpoint& checkpoint, const SessionState& session,
                                      const ModelGeometry& g, std::string& error);
bool conversation_checkpoint_save(ConversationCheckpoint& checkpoint, const SessionState& session,
                                  const ModelGeometry& g, std::string& error);
bool conversation_checkpoint_restore(const ConversationCheckpoint& checkpoint, SessionState& session,
                                     const ModelGeometry& g, std::string& error);
// conversation_checkpoint_save(from) + conversation_checkpoint_restore(to) of `ids` without a host image, for two
// sessions carved alike on the same device; the caller synchronizes the device before (as for the restore).
// `queued`: the copies are only queued (the private copy stream), and the caller synchronizes the device after too.
bool conversation_checkpoint_copy(const SessionState& from, SessionState& to, const std::vector<int32_t>& ids,
                                  const ModelGeometry& g, std::string& error, bool queued = false);

struct ConversationView {
    const std::vector<int32_t>& ids;
    const std::vector<ConversationImageKey>& images;
    const std::vector<ConversationCheckpoint>& checkpoints;
    bool cvec;
};
bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& session,
                                 const ModelGeometry& g, const QsaState& draft, size_t& bytes, std::string& error);
bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& session, const ModelGeometry& g,
                                         const QsaState& draft, size_t& bytes, std::string& error);
// The capture estimate includes retained capacity and transient segment directories;
// only estimate - reuse.bytes() requires additional physical RAM. Capture consumes
// the uniquely owned reusable buffers, including on failure.
// Caller admits the estimate before invoking capture. Allocation failures propagate
// to the RAM policy; the active session is never modified by capture.
// `live_from` (PLAN-p124a2, manager review 2026-11-26): take the running state as a host copy of this
// checkpoint - the turn capture's own host copy of it, for the same tokens - instead of reading it off the
// device, and refuse (false, nothing captured) if its arrays do not match this session's. The K/V is still
// captured normally. The caller has established the session has not run since that copy (park::reuse_turn_state).
bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view,
                                const SessionState& session, const ModelGeometry& g,
                                const QsaState& draft, std::string& error,
                                ConversationKvReuse reuse = {}, size_t* reused_bytes = nullptr,
                                const ConversationCheckpoint* live_from = nullptr);
// Disk save without capturing the K/V on the host: `meta` gets everything but the K/V (running state copied,
// checkpoints as given by the view), `sources` one streamed source per QSA layer then the draft.  Caller has
// synchronized and must not run the session until the file is written.
bool conversation_snapshot_sources(SavedConversation& meta, std::vector<SessionKvSource>& sources,
                                   const ConversationView& view, const SessionState& session,
                                   const ModelGeometry& g, const QsaState& draft, std::string& error);
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& session,
                                    const ModelGeometry& g, const QsaState& draft, std::string& error);
/// #2409: true when a restore (or a copy) on CUDA device `device` from the calling thread creates no copy stream: this
/// thread has one for the card or a finished thread left one in the pool. A restore writes into buffers that exist
/// already; a new stream is the one thing it can allocate (see program/conv_cache.hpp restore_fits).
bool conversation_copy_stream_ready(int device);
enum class ConversationRestore { restored, invalid, transfer_failed };
// Invalid images are rejected before any CUDA call/write. Transfer failure may
// leave partial state: caller MUST NOT continue inference from that session.
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& session,
                                                   const ModelGeometry& g, const QsaState& draft,
                                                   std::string& error);
// The K/V of the image's first `upto` tokens (main layers and the draft layer), not its running state: the caller
// mounts a checkpoint taken at `upto`.  For a checkpoint of a parked conversation that a new conversation starts
// with, while the parked one stays where it is (ConversationCache::borrows).  Same failure contract as above.
ConversationRestore conversation_snapshot_restore_prefix(const SavedConversation& image, int64_t upto,
                                                          SessionState& session, const ModelGeometry& g,
                                                          const QsaState& draft, std::string& error);

// The same with `draft == nullptr`: an image WITHOUT the draft layer's K/V (kv holds the session's own QSA layers
// only).  A layer split's later stages park this way; the draft ring is saved once, with the first stage's image.
// An image is validated and restored with the same kind of call it was saved with (a K/V layer count mismatch is
// rejected as invalid).
bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& session, const ModelGeometry& g,
                                 const QsaState* draft, size_t& bytes, std::string& error);
bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& session, const ModelGeometry& g, const QsaState* draft,
                                         size_t& bytes, std::string& error);
bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view, const SessionState& session,
                                const ModelGeometry& g, const QsaState* draft, std::string& error,
                                ConversationKvReuse reuse = {}, size_t* reused_bytes = nullptr,
                                const ConversationCheckpoint* live_from = nullptr);
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& session,
                                    const ModelGeometry& g, const QsaState* draft, std::string& error);
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& session,
                                                   const ModelGeometry& g, const QsaState* draft, std::string& error);
ConversationRestore conversation_snapshot_restore_prefix(const SavedConversation& image, int64_t upto,
                                                          SessionState& session, const ModelGeometry& g,
                                                          const QsaState* draft, std::string& error);
// Disk sessions with a layer split: the same with `draft == nullptr` for a stage without the draft layer (its file
// image then holds its own QSA layers' K/V only).
bool conversation_snapshot_sources(SavedConversation& meta, std::vector<SessionKvSource>& sources,
                                   const ConversationView& view, const SessionState& session,
                                   const ModelGeometry& g, const QsaState* draft, std::string& error);
// PLAN-p124a2: a park's image built by ADOPTING the whole-capture K/V a turn save (session_log_capture) already
// holds: every part aliases one block of the capture's host buffer (`turn`, kept alive by the returned image),
// so the K/V is not copied a second time and the device is not touched for it. The running state is copied from
// the capture's host copy and the checkpoint tree from the view, as conversation_snapshot_save would. The bytes
// are the same ones conversation_snapshot_save would capture for the same session at the same token count -
// the caller has established that nothing has rewritten the session since the capture (park::adopt_turn_capture).
// Returns false, leaving `image` empty, when the capture does not line up with this session (multi-stage, not
// whole, a part split across segments, any extent mismatch): the caller captures normally.
// `aliased_bytes`: the K/V bytes the image adopts instead of allocating (output only).
bool conversation_snapshot_from_capture(SavedConversation& image, const ConversationView& view,
                                        const SessionState& session, const ModelGeometry& g, const QsaState* draft,
                                        std::shared_ptr<const SessionLogTurn> turn, std::string& error,
                                        size_t* aliased_bytes = nullptr);
// The K/V bytes a whole capture of this session at `upto` tokens holds: every part of every owned layer plus the
// draft (0 when the layouts refuse). A park that adopts a capture subtracts this from its allocation estimate.
size_t conversation_snapshot_kv_bytes(const SessionState& session, const ModelGeometry& g, const QsaState* draft,
                                      int64_t upto);
bool conversation_session_read_limits(SessionReadLimits& limits, const SessionState& session, const ModelGeometry& g,
                                      const QsaState* draft, uint64_t max_tokens, uint64_t max_checkpoints,
                                      std::string& error);

} // namespace strata::core
