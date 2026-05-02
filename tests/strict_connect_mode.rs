//! Tests for strict connect-mode enforcement.
//!
//! When `MicroResolveConfig::server` is `Some(...)`, every mutation method on
//! `NamespaceHandle` must return `Err(Error::ConnectMode)`. Reads and
//! background sync must continue to work normally.
//!
//! Three tests:
//!   1. standalone — mutations succeed
//!   2. connected — mutations refused with `Error::ConnectMode`
//!   3. sync thread — background apply_ops path is NOT blocked by the guard

mod common;
#[allow(unused_imports)]
use common::*;
use serde_json::json;
use std::time::Duration;

const NS: &str = "strict-connect-mode-e2e";

fn wait_tick() {
    std::thread::sleep(Duration::from_secs(3));
}

/// 1. Standalone-mode mutations succeed.
#[test]
fn standalone_mutations_ok() {
    let engine = microresolve::MicroResolve::new(microresolve::MicroResolveConfig::default())
        .expect("standalone engine");
    let ns = engine.namespace("test");

    // All mutation methods must succeed.
    ns.add_intent("greet", &["hello world"][..])
        .expect("add_intent ok");
    ns.add_phrase("greet", "hi there", "en")
        .expect("add_phrase ok");
    ns.remove_phrase("greet", "hi there")
        .expect("remove_phrase ok");
    ns.set_domain_description("greet-domain", "Greeting domain")
        .expect("set_domain_description ok");
    ns.remove_domain_description("greet-domain")
        .expect("remove_domain_description ok");
    ns.reinforce_tokens(&["hello"], "greet")
        .expect("reinforce_tokens ok");
    ns.apply_weight_updates(&[("hello".to_string(), "greet".to_string(), 0.9)])
        .expect("apply_weight_updates ok");
    ns.decay_for_intents(&["hello world".to_string()], &["greet".to_string()], 0.1)
        .expect("decay_for_intents ok");
    ns.index_phrase("greet", "howdy").expect("index_phrase ok");
    ns.rebuild_caches().expect("rebuild_caches ok");
    ns.rebuild_index().expect("rebuild_index ok");
    ns.apply_review(&Default::default(), &[], &[], "hello", 0.1)
        .expect("apply_review ok");
    ns.remove_intent("greet").expect("remove_intent ok");
    assert_eq!(ns.intent_count(), 0, "intent removed");
}

/// 2. Connected-mode mutations are refused with `Error::ConnectMode`.
#[test]
fn connected_mutations_refused() {
    // Spawn a real server and build a connected engine pointing at it.
    let server = TestServer::spawn();
    let c = server.client();
    let b = format!("{}/api", server.url);
    let h = vec![("X-Namespace-ID", NS)];

    // Seed two intents — IDF requires N >= 2 to produce non-zero scores.
    let (s, body) = post_json(
        &c,
        &format!("{}/namespaces", b),
        &[],
        &json!({"namespace_id": NS}),
    );
    assert!((200..300).contains(&s), "create namespace: {}", body);
    post_json(
        &c,
        &format!("{}/intents", b),
        &h,
        &json!({"id": "greet", "phrases": ["hello", "hi"]}),
    );
    post_json(
        &c,
        &format!("{}/intents", b),
        &h,
        &json!({"id": "farewell", "phrases": ["goodbye", "bye"]}),
    );

    let engine = microresolve::MicroResolve::new(microresolve::MicroResolveConfig {
        server: Some(microresolve::ServerConfig {
            url: server.url.clone(),
            api_key: Some(server.api_key.clone()),
            subscribe: vec![NS.to_string()],
            tick_interval_secs: 1,
            log_buffer_max: 100,
        }),
        ..Default::default()
    })
    .expect("connected engine");

    let ns = engine.namespace(NS);

    // Verify the engine has the namespace populated (basic sanity).
    assert!(
        ns.intent_count() >= 1,
        "should have >=1 intent after initial sync"
    );

    // Every mutation method must fail with ConnectMode.
    macro_rules! assert_connect_mode {
        ($name:expr, $expr:expr) => {{
            match $expr {
                Err(microresolve::Error::ConnectMode) => {} // expected
                Err(other) => panic!("{}: expected ConnectMode, got: {}", $name, other),
                Ok(_) => panic!(
                    "{}: expected ConnectMode error, but mutation succeeded",
                    $name
                ),
            }
        }};
    }

    assert_connect_mode!(
        "add_intent",
        ns.add_intent("new-intent", &["some phrase"][..])
    );
    assert_connect_mode!("remove_intent", ns.remove_intent("greet"));
    assert_connect_mode!("add_phrase", ns.add_phrase("greet", "howdy", "en"));
    assert_connect_mode!("remove_phrase", ns.remove_phrase("greet", "hello"));
    assert_connect_mode!(
        "set_domain_description",
        ns.set_domain_description("x", "desc")
    );
    assert_connect_mode!(
        "remove_domain_description",
        ns.remove_domain_description("x")
    );
    assert_connect_mode!("reinforce_tokens", ns.reinforce_tokens(&["hello"], "greet"));
    assert_connect_mode!(
        "apply_weight_updates",
        ns.apply_weight_updates(&[("hello".to_string(), "greet".to_string(), 0.5)])
    );
    assert_connect_mode!(
        "decay_for_intents",
        ns.decay_for_intents(&["hello".to_string()], &["greet".to_string()], 0.1)
    );
    assert_connect_mode!("index_phrase", ns.index_phrase("greet", "hey there"));
    assert_connect_mode!("rebuild_caches", ns.rebuild_caches());
    assert_connect_mode!("rebuild_index", ns.rebuild_index());
    assert_connect_mode!(
        "apply_review",
        ns.apply_review(&Default::default(), &[], &[], "q", 0.1)
    );
    assert_connect_mode!("correct", ns.correct("hello", "greet", "greet2"));

    // Reads must still work. Use route_multi with a low threshold —
    // resolve()'s default threshold (0.3) is just above what a single
    // seed-phrase produces (~0.28 with IDF=ln(2) and weight=0.4).
    let result = ns.route_multi("hello", None, 0.0, 0.05);
    assert!(
        !result.multi.is_empty(),
        "read methods must still work in connected mode; got: {:?}",
        result.multi
    );
    let _ = ns.intent_ids();
    let _ = ns.intent_count();
    let _ = ns.version();

    // Cleanup.
    delete_json(
        &c,
        &format!("{}/namespaces", b),
        &[],
        &json!({"namespace_id": NS}),
    );
}

/// 3. Background sync thread applies ops even though local mutations are blocked.
///
/// The sync thread directly holds `namespaces.write()` — it does NOT go through
/// `with_resolver_mut` — so it must not be affected by the ConnectMode guard.
#[test]
fn sync_thread_works_in_connect_mode() {
    const NS2: &str = "strict-connect-sync-thread";
    let server = TestServer::spawn();
    let c = server.client();
    let b = format!("{}/api", server.url);
    let h = vec![("X-Namespace-ID", NS2)];

    // Seed namespace on server.
    post_json(
        &c,
        &format!("{}/namespaces", b),
        &[],
        &json!({"namespace_id": NS2}),
    );
    post_json(
        &c,
        &format!("{}/intents", b),
        &h,
        &json!({"id": "greet", "phrases": ["hello", "hi"]}),
    );
    post_json(
        &c,
        &format!("{}/intents", b),
        &h,
        &json!({"id": "farewell", "phrases": ["goodbye", "bye"]}),
    );

    // Boot connected engine.
    let engine = microresolve::MicroResolve::new(microresolve::MicroResolveConfig {
        server: Some(microresolve::ServerConfig {
            url: server.url.clone(),
            api_key: Some(server.api_key.clone()),
            subscribe: vec![NS2.to_string()],
            tick_interval_secs: 1,
            log_buffer_max: 100,
        }),
        ..Default::default()
    })
    .expect("connected engine for sync thread test");

    assert!(
        engine.namespace(NS2).intent_count() >= 1,
        "initial sync loaded at least one intent"
    );

    // Server-side mutation: add a phrase. This must be propagated to the
    // connected engine via the background sync thread.
    let (s, _) = post_json(
        &c,
        &format!("{}/intents/greet/phrases", b),
        &h,
        &json!({"phrase": "howdy partner", "lang": "en"}),
    );
    assert!((200..300).contains(&s), "server-side add phrase");

    // Wait for the background thread to tick.
    wait_tick();

    // The new phrase should now be routable.
    let result = engine
        .namespace(NS2)
        .route_multi("howdy partner", None, 0.05, 0.0);
    assert!(
        result.multi.iter().any(|(id, _)| id == "greet"),
        "background sync applied the server-side phrase; got: {:?}",
        result.multi
    );

    // Cleanup.
    delete_json(
        &c,
        &format!("{}/namespaces", b),
        &[],
        &json!({"namespace_id": NS2}),
    );
}
