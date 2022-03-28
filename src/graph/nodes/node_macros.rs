macro_rules! node_ext {
    ($($variant:ident),+ $(,)?) => {
        impl NodeExt for Node {
            fn node(&self) -> NodeId {
                match self {
                    $(Self::$variant(node, ..) => node.node(),)+
                }
            }

            fn input_desc(&self) -> EdgeDescriptor {
                match self {
                    $(Self::$variant(node, ..) => node.input_desc(),)+
                }
            }

            fn all_input_ports(&self) -> InputPorts {
                match self {
                    $(Self::$variant(node, ..) => node.all_input_ports(),)+
                }
            }

            fn all_input_port_kinds(&self) -> InputPortKinds {
                match self {
                    $(Self::$variant(node, ..) => node.all_input_port_kinds(),)+
                }
            }

            fn update_input(&mut self, from: InputPort, to: InputPort) {
                match self {
                    $(Self::$variant(node, ..) => node.update_input(from, to),)+
                }
            }

            fn output_desc(&self) -> EdgeDescriptor {
                match self {
                    $(Self::$variant(node, ..) => node.output_desc(),)+
                }
            }

            fn all_output_ports(&self) -> OutputPorts {
                match self {
                    $(Self::$variant(node, ..) => node.all_output_ports(),)+
                }
            }

            fn all_output_port_kinds(&self) -> OutputPortKinds {
                match self {
                    $(Self::$variant(node, ..) => node.all_output_port_kinds(),)+
                }
            }

            fn update_output(&mut self, from: OutputPort, to: OutputPort) {
                match self {
                    $(Self::$variant(node, ..) => node.update_output(from, to),)+
                }
            }
        }
    };
}

macro_rules! node_methods {
    ($($type:ident $(as $name:ident)?),* $(,)?) => {
        use paste::paste;

        $(
            impl Node {
                paste! {
                    pub const fn [<is_ $type:snake>](&self) -> bool {
                        matches!(self, node_methods!(@pat $type, $($name)?))
                    }

                    pub fn [<as_ $type:snake>](&self) -> Option<&$type> {
                        if let node_methods!(@variant node, $type, $($name)?) = self {
                            Some(node)
                        } else {
                            None
                        }
                    }

                    pub fn [<as_ $type:snake _mut>](&mut self) -> Option<&mut $type> {
                        if let node_methods!(@variant node, $type, $($name)?) = self {
                            Some(node)
                        } else {
                            None
                        }
                    }

                    #[track_caller]
                    pub fn [<to_ $type:snake>](&self) -> &$type {
                        if let node_methods!(@variant node, $type, $($name)?) = self {
                            node
                        } else {
                            panic!(
                                concat!("attempted to get", stringify!($type), " got {:?}"),
                                self,
                            );
                        }
                    }

                    #[track_caller]
                    pub fn [<to_ $type:snake _mut>](&mut self) -> &mut $type {
                        if let node_methods!(@variant node, $type, $($name)?) = self {
                            node
                        } else {
                            panic!(
                                concat!("attempted to get", stringify!($type), " got {:?}"),
                                self,
                            );
                        }
                    }
                }
            }
        )*
    };

    (@variant $inner:ident, $variant_type:ident, $variant_name:ident $(,)?) => { Node::$variant_name($inner, ..) };
    (@variant $inner:ident, $variant_type:ident $(,)?) => { Node::$variant_type($inner, ..) };

    (@pat $variant_type:ident, $variant_name:ident $(,)?) => { Node::$variant_name(..) };
    (@pat $variant_type:ident $(,)?) => { Node::$variant_type(..) };
}

macro_rules! node_conversions {
    ($($type:ident $(as $name:ident)?),* $(,)?) => {
        $(
            impl From<$type> for Node {
                fn from(node: $type) -> Self {
                    node_conversions!(@variant node, $type, $($name)?)
                }
            }

            impl TryInto<$type> for Node {
                type Error = Self;

                fn try_into(self) -> Result<$type, Self::Error> {
                    if let node_conversions!(@variant node, $type, $($name)?) = self {
                        Ok(node)
                    } else {
                        Err(self)
                    }
                }
            }

            impl TryInto<$type> for &Node {
                type Error = Self;

                fn try_into(self) -> Result<$type, Self::Error> {
                    if let node_conversions!(@variant node, $type, $($name)?) = self {
                        Ok(node.clone())
                    } else {
                        Err(self)
                    }
                }
            }

            impl<'a> TryInto<&'a $type> for &'a Node {
                type Error = Self;

                fn try_into(self) -> Result<&'a $type, Self::Error> {
                    if let node_conversions!(@variant node, $type, $($name)?) = self {
                        Ok(node)
                    } else {
                        Err(self)
                    }
                }
            }
        )*
    };

    (@variant $inner:ident, $variant_type:ident, $variant_name:ident $(,)?) => { Node::$variant_name($inner, ..) };
    (@variant $inner:ident, $variant_type:ident $(,)?) => { Node::$variant_type($inner) };

    (@pat $variant_type:ident, $variant_name:ident $(,)?) => { Node::$variant_name(..) };
    (@pat $variant_type:ident $(,)?) => { Node::$variant_type(..) };
}
