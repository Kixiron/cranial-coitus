pub(crate) trait AssertNone {
    fn unwrap_none(&self);

    #[track_caller]
    fn debug_unwrap_none(&self) {
        if cfg!(debug_assertions) {
            self.unwrap_none();
        }
    }

    fn expect_none(&self, message: &str);

    #[track_caller]
    fn debug_expect_none(&self, message: &str) {
        if cfg!(debug_assertions) {
            self.expect_none(message);
        }
    }
}

impl<T> AssertNone for Option<T> {
    #[track_caller]
    fn unwrap_none(&self) {
        if self.is_some() {
            panic!("unwrapped a `Some` value where a `None` was expected");
        }
    }

    #[track_caller]
    fn expect_none(&self, message: &str) {
        if self.is_some() {
            panic!(
                "unwrapped a `Some` value where a `None` was expected: {}",
                message,
            );
        }
    }
}
