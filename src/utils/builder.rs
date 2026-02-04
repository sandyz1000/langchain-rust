/// Builder trait definition
/// Provides a unified interface for all Builder types.
pub trait Builder<T> {
    /// Build the final object
    fn build(self) -> Result<T, Box<dyn std::error::Error>>;
}

/// Builder trait with validation
/// Supports validation in the Builder trait.
pub trait ValidatedBuilder<T> {
    /// Validate build parameters
    fn validate(&self) -> Result<(), String>;

    /// Build the final object (with validation)
    fn build(self) -> Result<T, Box<dyn std::error::Error>>
    where
        Self: Sized,
    {
        self.validate()?;
        self.build_unchecked()
    }

    /// Build the final object (without validation)
    fn build_unchecked(self) -> Result<T, Box<dyn std::error::Error>>;
}

/// Macro: Create simple Builder structures
// / /// # example
// / /// ```rust,ignore
// / use langchain_ai_rust::utils::simple_builder;
// / /// simple_builder! {
// / pub struct MyBuilder {
// / field1: Option<String>,
// / field2: Option<i32>,
// / }
// / impl {
// / pub fn with_field1(mut self, value: String) -> Self {
// / self.field1 = Some(value);
// / self
// / }
// / pub fn with_field2(mut self, value: i32) -> Self {
// / self.field2 = Some(value);
// / self
// / }
// / }
// / }
// / ```
#[macro_export]
macro_rules! simple_builder {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident {
            $($field:ident: $type:ty),* $(,)?
        }
        impl {
            $($method:item)*
        }
    ) => {
        $(#[$meta])*
        $vis struct $name {
            $($field: $type),*
        }

        impl $name {
            pub fn new() -> Self {
                Self {
                    $($field: None),*
                }
            }

            $($method)*
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestStruct {
        value: String,
    }

    struct TestBuilder {
        value: Option<String>,
    }

    impl TestBuilder {
        fn new() -> Self {
            Self { value: None }
        }

        fn with_value(mut self, value: String) -> Self {
            self.value = Some(value);
            self
        }
    }

    impl Builder<TestStruct> for TestBuilder {
        fn build(self) -> Result<TestStruct, Box<dyn std::error::Error>> {
            Ok(TestStruct {
                value: self.value.unwrap_or_else(|| "default".to_string()),
            })
        }
    }

    #[test]
    fn test_builder() {
        let builder = TestBuilder::new().with_value("test".to_string());
        let result = builder.build().unwrap();
        assert_eq!(result.value, "test");
    }
}
